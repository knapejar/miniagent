# -*- coding: utf-8 -*-
"""Web access: keyless search and page extraction, standard library only."""
import gzip
import json
import os
import re
import urllib.error
import urllib.parse
import time
import urllib.request
import zlib
from html.parser import HTMLParser

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TIMEOUT = 25
SEARCH_TIMEOUT = 8      # an engine that has not answered by now will not help
COOLDOWN = 600         # seconds a failed provider is skipped for
MAX_BYTES = 3 * 1024 * 1024
SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer",
             "aside", "form", "iframe", "template"}
BLOCK_TAGS = {"p", "div", "section", "article", "br", "li", "tr", "table",
              "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class _Redirects(urllib.request.HTTPRedirectHandler):
    """urllib refuses 307/308 and dies on a redirect without a Location."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not headers.get("Location"):
            return None
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl)

    http_error_307 = urllib.request.HTTPRedirectHandler.http_error_302
    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302


_opener = urllib.request.build_opener(_Redirects)
_cooldown = {}          # provider -> time it may be tried again


def get(url, data=None, headers=None, timeout=None):
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("User-Agent", UA)
    request.add_header("Accept-Language", "en;q=0.9")
    request.add_header("Accept-Encoding", "gzip, deflate")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with _opener.open(request, timeout=timeout or TIMEOUT) as response:
        raw = response.read(MAX_BYTES)
        encoding = response.headers.get("Content-Encoding", "")
        if "gzip" in encoding:
            raw = gzip.decompress(raw)
        elif "deflate" in encoding:
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, "replace"), response.headers.get_content_type()


class TextExtractor(HTMLParser):
    """Readable text from HTML: drops chrome, keeps headings and paragraphs."""

    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.parts = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in HEADINGS:
            self.parts.append("\n\n" + HEADINGS[tag] + " ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS or tag in HEADINGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data.strip()
        elif data.strip():
            self.parts.append(data)

    def text(self):
        body = "".join(self.parts)
        body = re.sub(r"[ \t\r\f\v]+", " ", body)
        body = re.sub(r" ?\n ?", "\n", body)
        return re.sub(r"\n{3,}", "\n\n", body).strip()


def extract(html):
    parser = TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.title, parser.text()


class _Links(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            title = " ".join("".join(self._text).split())
            if title:
                self.links.append((title, _unwrap(self._href)))
            self._href = None


def _unwrap(href):
    if "duckduckgo.com/l/" in href or href.startswith("/l/"):
        query = urllib.parse.urlparse(href).query
        target = urllib.parse.parse_qs(query).get("uddg")
        if target:
            return target[0]
    return href


def _is_result(url, engine_host):
    return (url.startswith("http")
            and engine_host not in url
            and not url.endswith((".css", ".js", ".ico"))
            and "/privacy" not in url and "/settings" not in url)


MIN_RESULTS = 3          # a real result page always has more than a couple


def _scrape(url, engine_host, limit, data=None):
    html, _ = get(url, data=data, timeout=SEARCH_TIMEOUT)
    parser = _Links()
    parser.feed(html)
    results, seen = [], set()
    for title, link in parser.links:
        if _is_result(link, engine_host) and link not in seen and len(title) > 4:
            seen.add(link)
            results.append({"title": title, "url": link, "snippet": ""})
        if len(results) >= limit:
            break
    # An anti-bot page still contains the engine's own footer links, so a
    # handful of hits means blocked, not answered. Treat it as a miss.
    return results if len(results) >= MIN_RESULTS else []


def searxng(query, limit):
    """A SearXNG running on localhost. It aggregates ~20 engines, rotates them
    and absorbs their blocks, so it is the only keyless route that survives an
    agent making several searches in a row."""
    base = os.getenv("MINIAGENT_SEARX", "http://127.0.0.1:8080")
    url = base.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"})
    raw, _ = get(url, headers={"Accept": "application/json"}, timeout=SEARCH_TIMEOUT)
    items = json.loads(raw).get("results", [])
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:200]}
            for r in items[:limit]]


def brave_html(query, limit):
    return _scrape("https://search.brave.com/search?" + urllib.parse.urlencode({"q": query}),
                   "brave.com", limit)


def mojeek(query, limit):
    return _scrape("https://www.mojeek.com/search?" + urllib.parse.urlencode({"q": query}),
                   "mojeek.com", limit)


def bing(query, limit):
    return _scrape("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query}),
                   "bing.com", limit)


def duckduckgo(query, limit):
    return _scrape("https://lite.duckduckgo.com/lite/", "duckduckgo.com", limit,
                   data=urllib.parse.urlencode({"q": query}).encode())


def duckduckgo_html(query, limit):
    return _scrape("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}),
                   "duckduckgo.com", limit)


def brave(query, limit, key):
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": limit})
    raw, _ = get(url, headers={"X-Subscription-Token": key, "Accept": "application/json"},
                 timeout=SEARCH_TIMEOUT)
    data = json.loads(raw)
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": re.sub(r"<[^>]+>", "", r.get("description", ""))}
            for r in (data.get("web") or {}).get("results", [])[:limit]]


def tavily(query, limit, key):
    payload = json.dumps({"api_key": key, "query": query, "max_results": limit}).encode()
    raw, _ = get("https://api.tavily.com/search", data=payload,
                 headers={"Content-Type": "application/json"}, timeout=SEARCH_TIMEOUT)
    data = json.loads(raw)
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:300]}
            for r in data.get("results", [])[:limit]]


def search(query, limit=6):
    """Walk the provider chain until one actually returns results.

    A single engine is not enough: DuckDuckGo starts answering with an
    anti-bot page once it has seen a few automated requests, and an engine
    that returns a page with no parseable results is a failure, not a hit.
    """
    providers = []
    if os.getenv("BRAVE_API_KEY"):
        providers.append(("brave", lambda: brave(query, limit, os.environ["BRAVE_API_KEY"])))
    if os.getenv("TAVILY_API_KEY"):
        providers.append(("tavily", lambda: tavily(query, limit, os.environ["TAVILY_API_KEY"])))
    providers += [("searxng", lambda: searxng(query, limit)),
                  ("brave-html", lambda: brave_html(query, limit)),
                  ("duckduckgo", lambda: duckduckgo(query, limit)),
                  ("duckduckgo-html", lambda: duckduckgo_html(query, limit)),
                  ("mojeek", lambda: mojeek(query, limit)),
                  ("bing", lambda: bing(query, limit))]

    errors, skipped = [], []
    now = time.time()
    for name, call in providers:
        if _cooldown.get(name, 0) > now:
            skipped.append(name)
            continue
        try:
            results = call()
            if results:
                _cooldown.pop(name, None)
                return name, results
            # An empty result set is a valid answer for that query, so it must
            # not put a working provider in the corner for ten minutes.
            errors.append("%s: no results" % name)
        except Exception as e:
            errors.append("%s: %s" % (name, str(e)[:80]))
            _cooldown[name] = time.time() + COOLDOWN
    if skipped and not errors:
        # Everything is cooling down; clear it rather than answer nothing at all.
        _cooldown.clear()
        return search(query, limit)
    raise RuntimeError("; ".join(errors) + (" (skipped: %s)" % ", ".join(skipped)
                                            if skipped else ""))
