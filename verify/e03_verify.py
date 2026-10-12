# -*- coding: utf-8 -*-
"""e03: edit and write keep a file's CRLF line endings, with a test."""
import os
import re
import tempfile

from common import check, load, report, source, suite_passes

files = load("agent.tools.files")
root = tempfile.mkdtemp()


class Ctx(object):
    cwd = root


def edited(raw, old, new):
    path = os.path.join(root, "f.txt")
    with open(path, "wb") as fh:
        fh.write(raw)
    files.EditTool().run(Ctx(), path=path, old=old, new=new)
    with open(path, "rb") as fh:
        return fh.read()


def written(raw, text, name="w.txt"):
    path = os.path.join(root, name)
    if raw is not None:
        with open(path, "wb") as fh:
            fh.write(raw)
    files.WriteTool().run(Ctx(), path=path, text=text)
    with open(path, "rb") as fh:
        return fh.read()


CRLF = b"one\r\ntwo\r\nthree\r\n"
check("edit", "a one-line edit keeps CRLF",
      lambda: edited(CRLF, "two", "TWO"), b"one\r\nTWO\r\nthree\r\n")
check("edit", "a multi-line edit keeps CRLF",
      lambda: edited(CRLF, "two\nthree", "2\n3\n3b"), b"one\r\n2\r\n3\r\n3b\r\n")
check("edit", "an LF file stays LF",
      lambda: edited(b"one\ntwo\n", "two", "TWO"), b"one\nTWO\n")
check("write", "overwriting a CRLF file keeps CRLF",
      lambda: written(CRLF, "a\nb\n", "crlf.txt"), b"a\r\nb\r\n")
check("write", "a new file is LF", lambda: written(None, "a\nb\n", "new.txt"), b"a\nb\n")
check("write", "overwriting an LF file stays LF",
      lambda: written(b"x\ny\n", "a\nb\n", "lf.txt"), b"a\nb\n")
check("test", "tests/test_agent.py tests CRLF",
      lambda: bool(re.search(r"crlf|\\r\\n", source(os.path.join("tests", "test_agent.py")), re.I)
                   and re.search(r"def test_\w*(crlf|line_end|newline)\w*",
                                 source(os.path.join("tests", "test_agent.py")), re.I)))
suite_passes()
report()
