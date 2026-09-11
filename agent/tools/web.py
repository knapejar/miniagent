# -*- coding: utf-8 -*-
"""Web search and browsing.

Both tools take several targets at once and run them in parallel. Measured on
four fresh queries: 2.8s one at a time, 1.0s together. The larger saving is on
the model side - it spends 3-4s generating each separate call, so four separate
calls (~17s) collapse into one (~5s).
"""
from .. import web
from ..jobs import parallel
from ..page import Page

DEFAULT_RESULTS = 6
MAX_RAW_CHARS = 6000
MAX_TARGETS = 6
class _WebTool(object):
    params = {}
    required = ()
    danger = False

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return self.danger

    @staticmethod
    def targets(value):
        """One per line, so several can be asked for in a single call."""
        items = [part.strip() for part in str(value or "").splitlines() if part.strip()]
        return items[:MAX_TARGETS]

    @staticmethod
    def background(ctx, args, label, work):
        """Hand the work to the job runner if the caller asked for it."""
        forced = getattr(getattr(ctx, "cfg", None), "async_web", False)
        flag = str(args.get("background", "")).lower()
        if not forced and flag not in ("1", "true", "yes", "on"):
            return None
        runner = getattr(ctx, "jobs", None)
        if runner is None:
            return None
        job_id = runner.submit(label, work)
        return ("started %s in the background: %s\nKeep working - the result is "
                "delivered to you automatically as soon as it is ready."
                % (job_id, label))


class WebSearchTool(_WebTool):
    name = "websearch"
    description = ("Search the web. One query per line runs them all at once, much "
                   "faster than separate calls. background=true answers later.")
    params = {"query": {"type": "string", "description": "one per line"},
              "limit": {"type": "integer", "description": "results each, default 6"},
              "background": {"type": "string"}}
    required = ("query",)

    def summary(self, args):
        queries = self.targets(args.get("query"))
        head = queries[0][:60] if queries else ""
        return head + (" (+%d more)" % (len(queries) - 1) if len(queries) > 1 else "")

    def run(self, ctx, query=None, limit=DEFAULT_RESULTS, **kw):
        queries = self.targets(query)
        if not queries:
            return "error: missing arg 'query'"
        count = max(1, min(int(limit or DEFAULT_RESULTS), 10))

        def work():
            return self._search_all(queries, count)

        started = self.background(ctx, kw, "websearch: " + "; ".join(queries)[:80], work)
        return started or work()

    @staticmethod
    def _search_all(queries, count):
        def one(query):
            provider, results = web.search(query, count)
            lines = ["# %s  (via %s)" % (query, provider)]
            for i, item in enumerate(results, 1):
                lines.append("%d. %s\n   %s" % (i, item["title"][:120], item["url"]))
                if item["snippet"]:
                    lines.append("   %s" % item["snippet"][:200])
            return "\n".join(lines)
        return "\n\n".join(parallel(queries, one))


class BrowseTool(_WebTool):
    name = "browse"
    description = ("Open a page; one URL per line opens several at once. mode=outline "
                   "(default): title, headings, first links. mode=text with part=N reads "
                   "it a part at a time. mode=links: all absolute URLs. find=REGEX: "
                   "matching lines and their part. background=true answers later.")
    params = {"url": {"type": "string", "description": "one per line"},
              "mode": {"type": "string", "description": "outline | text | links"},
              "part": {"type": "integer"},
              "find": {"type": "string", "description": "regex"},
              "background": {"type": "string"}}
    required = ("url",)

    def summary(self, args):
        urls = self.targets(args.get("url"))
        extra = str(args.get("mode") or "outline")
        if args.get("part"):
            extra += " part=%s" % args["part"]
        if args.get("find"):
            extra += " find=%s" % str(args["find"])[:24]
        head = urls[0][:64] if urls else ""
        more = " (+%d more)" % (len(urls) - 1) if len(urls) > 1 else ""
        return "%s%s  [%s]" % (head, more, extra)

    def run(self, ctx, url=None, mode=None, part=1, find=None, **kw):
        urls = self.targets(url)
        if not urls:
            return "error: missing arg 'url'"

        def work():
            return "\n\n".join(parallel(urls, lambda u: self._one(u, mode, part, find)))

        started = self.background(ctx, kw, "browse: " + ", ".join(urls)[:80], work)
        return started or work()

    @staticmethod
    def _one(url, mode, part, find):
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            body, content_type = web.get(url)
        except Exception as e:
            return "%s\nerror: fetch failed (%s)" % (url, e)

        if "html" not in (content_type or ""):
            text = body[:MAX_RAW_CHARS]
            suffix = "" if len(body) <= MAX_RAW_CHARS else \
                "\n...[%d more chars]" % (len(body) - MAX_RAW_CHARS)
            return "%s\n%s%s" % (url, text, suffix)

        page = Page(body, url)
        if find:
            return page.search(find)
        mode = str(mode or "outline").lower()
        if mode.startswith("l"):
            return page.link_list()
        if mode.startswith("t"):
            return page.part(part or 1)
        return page.outline()
