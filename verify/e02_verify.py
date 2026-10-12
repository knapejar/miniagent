# -*- coding: utf-8 -*-
"""e02: grep gains a `context` argument (like grep -C), with a test."""
import os
import re
import tempfile

from common import check, load, report, source, suite_passes

GrepTool = load("agent.tools.search").GrepTool
root = tempfile.mkdtemp()
with open(os.path.join(root, "f.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join("line %d%s" % (i, " MATCH" if i in (5, 7) else "")
                       for i in range(1, 13)) + "\n")
with open(os.path.join(root, "g.txt"), "w", encoding="utf-8") as fh:
    fh.write("a\nb\nONE\nc\nd\ne\nf\n")


class Ctx(object):
    cwd = root


def grep(pattern, **kw):
    return GrepTool().run(Ctx(), pattern=pattern, **kw)


def lines(out):
    return [l for l in out.splitlines() if l.strip()]


check("spec", "context is one of the tool's params", lambda: "context" in GrepTool.params)
check("plain", "without context only the match is shown",
      lambda: lines(grep("ONE", path="g.txt")) == ["g.txt:3: ONE"])
check("context", "context=1 shows the line before",
      lambda: any(re.match(r"g\.txt-2-\s*b$", l) for l in lines(grep("ONE", path="g.txt", context=1))))
check("context", "context=1 shows the line after",
      lambda: any(re.match(r"g\.txt-4-\s*c$", l) for l in lines(grep("ONE", path="g.txt", context=1))))
check("context", "the match keeps path:LINE: text",
      lambda: "g.txt:3: ONE" in lines(grep("ONE", path="g.txt", context=1)))
check("context", "context=1 shows exactly three lines",
      lambda: len(lines(grep("ONE", path="g.txt", context=1))) == 3)
check("context", "context given as a string works too",
      lambda: len(lines(grep("ONE", path="g.txt", context="2"))) == 5)
check("overlap", "two close matches share their context, no line twice",
      lambda: [l.split()[0][:8] for l in lines(grep("MATCH", path="f.txt", context=1))]
      == ["f.txt-4-", "f.txt:5:", "f.txt-6-", "f.txt:7:", "f.txt-8-"])
check("test", "tests/test_agent.py has a grep test using context",
      lambda: bool(re.search(r"def test_\w*grep\w*context|def test_\w*context\w*grep"
                             r"|context\s*=\s*['\"]?\d", source(os.path.join("tests", "test_agent.py")))))
suite_passes()
report()
