# -*- coding: utf-8 -*-
"""Nezavisle overeni ulohy h08 (Markdown -> HTML).

Agent tenhle soubor nevidi - pracuje ve workspace/h08 a testy si pise sam.
Tady kontrolujeme, jestli to, co postavil, opravdu funguje.

Porovnavani je zamerne tolerantni: normalizuje se bily znak mezi tagy, protoze
zadani nepredepisuje presne odsazeni. Kontroluje se pritomnost ocekavaneho
fragmentu, ne bajtova shoda.

    python verify/h08_verify.py [cesta_k_md2html.py]
"""
import os
import re
import sys

WS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace", "h08")
sys.path.insert(0, os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else WS))

try:
    import md2html
except Exception as e:                                    # noqa: BLE001
    print("FATAL: md2html.py se nepodarilo importovat: %s" % e)
    sys.exit(2)

render = getattr(md2html, "render", None)
if not callable(render):
    print("FATAL: md2html.render(md) neexistuje")
    sys.exit(2)


def norm(h):
    """Sjednoti bily znak, aby na odsazeni nezalezelo."""
    h = re.sub(r">\s+<", "><", h.strip())
    return re.sub(r"\s+", " ", h)


RESULTS = []


def chk(group, md, expect, absent=None):
    """expect: fragment (nebo seznam), ktery MUSI byt ve vystupu."""
    try:
        out = norm(render(md))
    except Exception as e:                                # noqa: BLE001
        RESULTS.append((group, md, "VYJIMKA: %r" % e, False))
        return
    want = expect if isinstance(expect, list) else [expect]
    ok = all(norm(w) in out for w in want)
    if ok and absent:
        bad = absent if isinstance(absent, list) else [absent]
        ok = all(norm(b) not in out for b in bad)
    RESULTS.append((group, md, out, ok))


# --- nadpisy -----------------------------------------------------------------
chk("nadpisy", "# Title", "<h1>Title</h1>")
chk("nadpisy", "## Sub", "<h2>Sub</h2>")
chk("nadpisy", "###### Six", "<h6>Six</h6>")
chk("nadpisy", "### With *em*", ["<h3>", "<em>em</em>", "</h3>"])

# --- odstavce ----------------------------------------------------------------
chk("odstavce", "hello", "<p>hello</p>")
chk("odstavce", "one\n\ntwo", ["<p>one</p>", "<p>two</p>"])
chk("odstavce", "line one\nline two", "<p>line one line two</p>")

# --- inline ------------------------------------------------------------------
chk("inline", "**bold**", "<strong>bold</strong>")
chk("inline", "*italic*", "<em>italic</em>")
chk("inline", "`code`", "<code>code</code>")
chk("inline", "***both***", ["<strong>", "<em>", "both"])
chk("inline", "a **b** c *d* e", ["<strong>b</strong>", "<em>d</em>"])
chk("inline", "`a*b*c`", "<code>a*b*c</code>", absent="<em>b</em>")

# --- odkazy a obrazky --------------------------------------------------------
chk("odkazy", "[text](http://x.com)", '<a href="http://x.com">text</a>')
chk("odkazy", "![alt](img.png)", ['<img src="img.png"', 'alt="alt"'])
chk("odkazy", "see [a](u) and [b](v)", ['<a href="u">a</a>', '<a href="v">b</a>'])
chk("odkazy", "[**bold link**](u)", ["<a href=\"u\">", "<strong>bold link</strong>"])

# --- seznamy -----------------------------------------------------------------
chk("seznamy", "- a\n- b", ["<ul>", "<li>a</li>", "<li>b</li>", "</ul>"])
chk("seznamy", "- a\n  - a1\n  - a2\n- b",
    ["<ul>", "<li>a", "<ul>", "<li>a1</li>", "<li>a2</li>", "</ul>", "<li>b</li>"])
chk("seznamy", "1. one\n2. two", ["<ol>", "<li>one</li>", "<li>two</li>", "</ol>"])
chk("seznamy", "1. one\n2. two\n3. three", ["<ol>", "<li>three</li>"])
chk("seznamy", "- **b** item", ["<li>", "<strong>b</strong>"])

# --- citace ------------------------------------------------------------------
chk("citace", "> quoted", ["<blockquote>", "quoted", "</blockquote>"])
chk("citace", "> outer\n> > inner", ["<blockquote>", "<blockquote>", "inner"])
chk("citace", "> a *b*", ["<blockquote>", "<em>b</em>"])

# --- code fence --------------------------------------------------------------
chk("fence", "```\nx = 1\n```", ["<pre>", "<code>", "x = 1"])
chk("fence", "```python\nprint(1)\n```", ["<code", "python", "print(1)"])
chk("fence", "```\n# not a heading\n```", "# not a heading", absent="<h1>")
chk("fence", "```\n- not a list\n```", "- not a list", absent="<li>")
chk("fence", "```\n**not bold**\n```", "**not bold**", absent="<strong>")

# --- horizontalni cara -------------------------------------------------------
chk("hr", "---", "<hr")
chk("hr", "a\n\n---\n\nb", ["<p>a</p>", "<hr", "<p>b</p>"])

# --- escapovani --------------------------------------------------------------
chk("escape", "a < b", "&lt;")
chk("escape", "a > b", "&gt;")
chk("escape", "AT&T", "&amp;")
chk("escape", "<script>alert(1)</script>", ["&lt;script&gt;"], absent="<script>")
chk("escape", "`<b>`", ["<code>", "&lt;b&gt;"], absent="<code><b></code>")

# --- pasti ze zadani ---------------------------------------------------------
chk("pasti", "2 * 3 = 6", "2 * 3 = 6", absent="<em>")
chk("pasti", "**unclosed bold", "**unclosed bold", absent="<strong>")
chk("pasti", "[`x`](u)", ['<a href="u">', "<code>x</code>"])
chk("pasti", "para text\n- a\n- b", ["<p>para text</p>", "<li>a</li>"])
chk("pasti", "```\n> quote ### and [l](u)\n```",
    "> quote ### and [l](u)", absent=["<blockquote>", "<h3>"])
chk("pasti", "a_b_c", "a_b_c")

# --- kombinace ---------------------------------------------------------------
chk("kombinace", "# T\n\npara **b**\n\n- i1\n- i2\n\n> q\n\n```\ncode\n```",
    ["<h1>T</h1>", "<strong>b</strong>", "<li>i1</li>", "<blockquote>", "code"])
chk("kombinace", "## H\n1. [a](u)\n2. `c`",
    ["<h2>H</h2>", "<ol>", '<a href="u">a</a>', "<code>c</code>"])

# --- vysledek ----------------------------------------------------------------
groups = {}
for g, md, out, ok in RESULTS:
    groups.setdefault(g, [0, 0])
    groups[g][1] += 1
    groups[g][0] += 1 if ok else 0

print("=" * 72)
for g, (p, t) in groups.items():
    print("  %-10s %2d/%2d %s" % (g, p, t, "OK" if p == t else "<-- chyby"))
passed = sum(1 for r in RESULTS if r[3])
print("=" * 72)
print("CELKEM: %d/%d (%.0f %%)" % (passed, len(RESULTS), 100.0 * passed / len(RESULTS)))

fails = [r for r in RESULTS if not r[3]]
if fails:
    print("\n--- co selhalo (max 15) ---")
    for g, md, out, _ in fails[:15]:
        print("[%s] vstup: %r\n        vystup: %s\n" % (g, md, out[:160]))
sys.exit(0 if not fails else 1)
