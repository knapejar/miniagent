# -*- coding: utf-8 -*-
"""Turn an HTML page into something an agent can navigate.

A dump of the first few thousand characters is not browsing: there is no way to
see what a page holds, no way to reach the rest of it, and no way to follow a
link, which leaves the model guessing URLs. A page here is an outline, a set of
absolute links, and text split into addressable parts.
"""
import re
import urllib.parse
from html.parser import HTMLParser

SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer",
             "aside", "form", "iframe", "template", "button", "select"}
BLOCK_TAGS = {"p", "div", "section", "article", "br", "li", "tr", "table",
              "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dt", "dd"}
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
MAIN_TAGS = {"main", "article"}
MAIN_IDS = {"content", "main", "main-content", "mw-content-text", "readme",
            "repo-content-pjax-container"}
PART_CHARS = 4000
MIN_MAIN_CHARS = 400          # below this the main region was a false positive


class _Parser(HTMLParser):
    def __init__(self, base_url, only_main):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.base_url = base_url
        self.only_main = only_main
        self.title = ""
        self.parts = []
        self.headings = []
        self.links = []
        self._skip = 0
        self._in_title = False
        self._main_depth = 0
        self._depth = 0
        self._heading = None
        self._href = None
        self._anchor = []

    # ------------------------------------------------------------- helpers
    def _capturing(self):
        if self._skip:
            return False
        return self._main_depth > 0 if self.only_main else True

    def _emit(self, text):
        if self._capturing():
            self.parts.append(text)

    # -------------------------------------------------------------- events
    def handle_starttag(self, tag, attrs):
        self._depth += 1
        attributes = dict(attrs)
        if tag in MAIN_TAGS or attributes.get("id") in MAIN_IDS:
            if self._main_depth == 0:
                self._main_start = self._depth
            self._main_depth += 1
        if tag in SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag in HEADINGS:
            self._heading = [HEADINGS[tag], []]
            self._emit("\n\n")
        elif tag == "a":
            href = attributes.get("href")
            self._href = urllib.parse.urljoin(self.base_url, href) if href else None
            self._anchor = []
        elif tag in BLOCK_TAGS:
            self._emit("\n")
        elif tag == "img" and self._capturing():
            alt = attributes.get("alt")
            if alt:
                self._emit(" [image: %s] " % alt)

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in HEADINGS and self._heading:
            text = " ".join("".join(self._heading[1]).split())
            if text and self._capturing():
                self.headings.append((self._heading[0], text))
                self._emit("%s %s\n" % ("#" * self._heading[0], text))
            self._heading = None
        elif tag == "a":
            if self._href and self._capturing():
                text = " ".join("".join(self._anchor).split())
                if text and not self._href.startswith("javascript:"):
                    self.links.append((text[:90], self._href))
                self._emit(text)
            self._href = None
        elif tag in BLOCK_TAGS:
            self._emit("\n")
        if self._main_depth and self._depth <= getattr(self, "_main_start", 0):
            self._main_depth = 0
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()
            return
        if self._heading is not None:
            self._heading[1].append(data)
            return
        if self._href is not None:
            self._anchor.append(data)
            return
        if data.strip():
            self._emit(data)

    def text(self):
        body = "".join(self.parts)
        body = re.sub(r"[ \t\r\f\v]+", " ", body)
        body = re.sub(r" ?\n ?", "\n", body)
        return re.sub(r"\n{3,}", "\n\n", body).strip()


class Page(object):
    """A parsed page: outline, absolute links, and text in addressable parts."""

    def __init__(self, html, url):
        main = _Parser(url, only_main=True)
        full = _Parser(url, only_main=False)
        for parser in (main, full):
            try:
                parser.feed(html)
            except Exception:
                pass
        # Prefer the <main>/<article> region, but only when it really holds the
        # page. Sites that mark up navigation as <article> would otherwise win.
        chosen = main if len(main.text()) >= MIN_MAIN_CHARS else full
        self.url = url
        self.title = full.title or main.title
        self.body = chosen.text()
        self.headings = chosen.headings or full.headings
        self.links = _dedupe(full.links)
        self.parts = _split(self.body, PART_CHARS)

    def outline(self, link_count=10):
        out = ["%s\n%s" % (self.title or "(no title)", self.url),
               "%d chars | %d parts | %d headings | %d links"
               % (len(self.body), len(self.parts), len(self.headings), len(self.links))]
        if self.headings:
            out.append("\nOUTLINE")
            for level, text in self.headings[:30]:
                out.append("%s%s" % ("  " * (level - 1), text[:90]))
        if self.links:
            out.append("\nLINKS (%d of %d, use mode=links for all)"
                       % (min(link_count, len(self.links)), len(self.links)))
            for text, href in self.links[:link_count]:
                out.append("%s -> %s" % (text, href))
        out.append("\nUse mode=text with part=1..%d to read it." % max(1, len(self.parts)))
        return "\n".join(out)

    def part(self, number):
        if not self.parts:
            return "(page has no text)"
        index = max(1, min(int(number), len(self.parts)))
        return "%s\n%s\n\n[part %d/%d]" % (
            self.url, self.parts[index - 1], index, len(self.parts))

    def link_list(self, limit=80):
        if not self.links:
            return "(no links)"
        rows = ["%s -> %s" % (text, href) for text, href in self.links[:limit]]
        more = "" if len(self.links) <= limit else \
            "\n...[%d more links]" % (len(self.links) - limit)
        return "%d links on %s\n%s%s" % (len(self.links), self.url, "\n".join(rows), more)

    def search(self, pattern):
        try:
            regex = re.compile(pattern, re.I)
        except re.error as e:
            return "error: bad regular expression (%s)" % e
        hits = []
        for number, part in enumerate(self.parts, 1):
            for line in part.splitlines():
                if regex.search(line):
                    hits.append("[part %d] %s" % (number, line.strip()[:200]))
        if not hits:
            return "no line matched %r on %s (%d parts)" % (pattern, self.url, len(self.parts))
        return "\n".join(hits[:60])


def _dedupe(links):
    seen, out = set(), []
    for text, href in links:
        if href not in seen and href.startswith(("http://", "https://")):
            seen.add(href)
            out.append((text, href))
    return out


def _split(text, size):
    if not text:
        return []
    parts, current = [], []
    length = 0
    for line in text.split("\n"):
        if length + len(line) > size and current:
            parts.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts
