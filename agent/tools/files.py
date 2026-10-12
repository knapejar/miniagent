# -*- coding: utf-8 -*-
"""File access: read / write / edit.

`edit` is deliberately the simplest possible patch - an exact string
replacement. For a small model that is more reliable than a unified-diff
envelope, and the failure message ("'old' not found") is trivial to act on.

What it is forgiving about comes from the traces: the 4B model got 33 of 56
edits wrong, mostly by one space of indentation, then re-read the file to find
out why. So a miss that differs only in indentation is applied anyway, any
other miss shows the closest text in the file, and every edit answers with the
lines around it - the re-read is already in the answer.
"""
import codecs
import difflib
import json
import locale
import os
import re

from .. import paths

READ_DEFAULT_LINES = 200
# A read is never cut in the middle (the loop clips other tools at 3000 chars):
# a hole in the code is what the model then edits from memory. Instead it stops
# early on a line boundary and says where to continue.
READ_MAX_CHARS = 8000
READ_MAX_FILES = 4         # paths one read takes
READ_MULTI_MIN = 2000      # chars each of several files gets at least
OUTLINE_MAX_CHARS = 1600
OUTLINE_LINE = 90
DEFINITION = re.compile(
    r"\s*(?:(?:async\s+def|def|class|function|func|fn|impl|struct|interface|enum|trait|module)\b"
    r"|export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const)\b"
    r"|(?:public|private|protected|internal|static)\s[^=;]*\()")
HEADING = re.compile(r"#{1,6}\s")
SNIPPET_CONTEXT = 3        # lines shown around an edit
SNIPPET_MAX_LINES = 30
CLOSEST_MAX_LINES = 40     # longest 'old' compared window by window
SHRINK_MIN_BYTES = 1000    # below this a rewrite to half the size is normal


def numbered(lines, first):
    # A tab, not "| ": the model copied the space after the bar into 'old'.
    return "\n".join("%5d\t%s" % (first + i, line) for i, line in enumerate(lines))


def check(path, source):
    """Cheap syntax check so a broken write is caught now, not ten steps later."""
    if path.endswith(".py"):
        try:
            compile(source, path, "exec")
        except (SyntaxError, ValueError) as e:
            return "\nSYNTAX ERROR line %s: %s" % (getattr(e, "lineno", "?"),
                                                  getattr(e, "msg", e))
    elif path.endswith(".json"):
        try:
            json.loads(source)
        except ValueError as e:
            return "\nJSON ERROR: %s" % e
    return ""


def validate(path):
    try:
        return check(path, load(path)[0])
    except Exception:
        return ""


def load(path):
    """(text with \\n line ends, encoding, newline) of a file as it is on disk.

    A file is written back the way it was read: an edit once turned a CRLF file
    into LF, which git shows as every line changed, and a cp1250 file could not
    be edited at all."""
    with open(path, "rb") as fh:
        raw = fh.read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    first = "utf-8-sig" if raw.startswith(codecs.BOM_UTF8) else "utf-8"
    for encoding in (first, locale.getpreferredencoding(False), "latin-1"):
        try:
            return raw.decode(encoding).replace("\r\n", "\n"), encoding, newline
        except (UnicodeDecodeError, LookupError):
            continue


def save(path, text, encoding="utf-8", newline="\n", mode="wb"):
    if mode == "ab" and encoding == "utf-8-sig":
        encoding = "utf-8"                        # the BOM belongs at the start only
    with open(path, mode) as fh:
        fh.write(text.replace("\r\n", "\n").replace("\n", newline).encode(encoding))


def remember(ctx, target):
    """Keep what the file held before this change, so it can be taken back:
    edit undo=true takes back one change, /undo all of the task's. The file as it
    was before the task is also copied to disk - the model once overwrote a
    module with a fragment and committed it."""
    history = getattr(ctx, "undo", None)
    target = os.path.abspath(target)
    if history is None or os.path.normcase(target).startswith(os.path.normcase(paths.home())):
        return                                    # the agent's own notes are not undone
    data = None
    if os.path.isfile(target):
        with open(target, "rb") as fh:
            data = fh.read()
    stack = history.setdefault(target, [])
    backups = getattr(ctx, "backup_dir", None)
    if not stack and data is not None and backups:
        os.makedirs(backups, exist_ok=True)
        with open(os.path.join(backups, re.sub(r"[:\\/]+", "-", target).strip("-")), "wb") as fh:
            fh.write(data)
    stack.append(data)


def restore(path, data):
    """Put back what remember() kept; None means the file did not exist."""
    if data is None:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "wb") as fh:
        fh.write(data)


class _FileTool(object):
    params = {}
    required = ()
    danger = False

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return self.danger

    def summary(self, args):
        return ", ".join("%s=%s" % (k, str(v).replace("\n", " ")[:50])
                         for k, v in args.items())

    @staticmethod
    def resolve(ctx, path):
        return path if os.path.isabs(path) else os.path.join(ctx.cwd, path)


class ReadTool(_FileTool):
    name = "read"
    description = "Read text files with line numbers; one path per line reads several."
    params = {"path": {"type": "string"},
              "start": {"type": "integer", "description": "first line, default 1"},
              "count": {"type": "integer", "description": "number of lines, default 200"}}
    required = ("path",)
    max_chars = READ_MAX_CHARS + OUTLINE_MAX_CHARS + 400    # the loop's clip limit for this tool

    def run(self, ctx, path=None, start=1, count=READ_DEFAULT_LINES, **_):
        # Every file it wanted used to cost a turn of its own.
        wanted = [p.strip() for p in str(path or "").splitlines() if p.strip()]
        if len(wanted) <= 1:
            return self.one(ctx, path, start, count, READ_MAX_CHARS)
        budget = max(READ_MULTI_MIN, READ_MAX_CHARS // min(len(wanted), READ_MAX_FILES))
        out = ["=== %s\n%s" % (p, self.one(ctx, p, start, count, budget))
               for p in wanted[:READ_MAX_FILES]]
        if len(wanted) > READ_MAX_FILES:
            out.append("...[%d more files not read - at most %d per call]"
                       % (len(wanted) - READ_MAX_FILES, READ_MAX_FILES))
        return "\n\n".join(out)

    def one(self, ctx, path, start, count, budget):
        try:
            lines = load(self.resolve(ctx, path))[0].splitlines()
        except Exception as e:
            return "error: %s" % e
        first, wanted = max(1, int(start)), int(count)
        shown, size = [], 0
        for line in lines[first - 1:first - 1 + wanted]:
            size += len(line) + 7
            if shown and size > budget:
                break
            shown.append(line[:budget])
        last = first - 1 + len(shown)
        if first == 1 and last >= len(lines):
            return numbered(shown, first) or "(empty file)"
        more = "\n...[lines %d-%d of %d shown, continue with start=%d]" % (
            first, last, len(lines), last + 1) if last < len(lines) else ""
        return (numbered(shown, first) or "(empty range)") + more + outline(
            path, lines, first, last)


def outline(path, lines, first, last):
    """Where the definitions are in the lines a read did not show: a model
    paging through a file 200 lines at a time read one file six times in a row."""
    pattern = HEADING if str(path).lower().endswith(".md") else DEFINITION
    rows, size = [], 0
    for number, line in enumerate(lines, 1):
        if first <= number <= last or not pattern.match(line):
            continue
        row = "%5d\t%s" % (number, line.rstrip()[:OUTLINE_LINE])
        size += len(row) + 1
        if size > OUTLINE_MAX_CHARS:
            rows.append("   ...")
            break
        rows.append(row)
    return "\nOutline of the lines not shown:\n" + "\n".join(rows) if rows else ""


class WriteTool(_FileTool):
    name = "write"
    description = "Create or overwrite a text file; append=true adds to its end."
    params = {"path": {"type": "string"}, "text": {"type": "string"},
              "append": {"type": "boolean"}}
    required = ("path", "text")

    def summary(self, args):
        text = str(args.get("text", ""))
        mode = "  append" if _yes(args.get("append")) else ""
        return "%s  (%d bytes)%s" % (args.get("path", "?"), len(text.encode("utf-8")), mode)

    def run(self, ctx, path=None, text="", append=False, **_):
        try:
            target = self.resolve(ctx, path)
            size = len(text.encode("utf-8"))
            existing = os.path.getsize(target) if os.path.exists(target) else 0
            # an existing file keeps its line ends and encoding, a new one is UTF-8 + LF
            _, encoding, newline = load(target) if existing else ("", "utf-8", "\n")
            if _yes(append):
                remember(ctx, target)
                save(target, text, encoding, newline, "ab")
                # No syntax check: a file written in parts is broken until the last one.
                return "appended %d bytes to %s (now %d bytes)" % (
                    size, target, os.path.getsize(target))
            if not text and existing:
                return ("error: refusing to overwrite %s with empty content. If you meant "
                        "to clear it, delete it with sh first." % target)
            if existing >= SHRINK_MIN_BYTES and size < existing // 2:
                # A 9 KB module was once replaced by the 384-byte fragment the
                # model meant to change, then committed.
                return ("error: refusing to shrink %s from %d to %d bytes - that drops most "
                        "of the file. Change a part of it with edit. If you really mean to "
                        "replace it whole, delete it with sh first." % (target, existing, size))
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            remember(ctx, target)
            save(target, text, encoding, newline)
            return "wrote %d bytes to %s%s" % (size, target, validate(target))
        except Exception as e:
            return "error: %s" % e


class EditTool(_FileTool):
    name = "edit"
    description = ("Replace the one exact occurrence of old with new in a file; "
                   "undo=true takes its last change back.")
    params = {"path": {"type": "string"}, "old": {"type": "string"},
              "new": {"type": "string"}, "undo": {"type": "boolean"}}
    required = ("path", "old", "new")

    def summary(self, args):
        # Showing truncated old/new makes distinct edits look identical, which
        # hides thrashing. Show sizes and the first differing fragment instead.
        if _yes(args.get("undo")):
            return "%s  undo" % args.get("path", "?")
        old, new = str(args.get("old", "")), str(args.get("new", ""))
        head = old.strip().splitlines()[0][:40] if old.strip() else "(empty)"
        return "%s  %r  %d->%d chars" % (args.get("path", "?"), head, len(old), len(new))

    def run(self, ctx, path=None, old="", new="", undo=False, **_):
        try:
            target = self.resolve(ctx, path)
            if _yes(undo):
                return self.undo(ctx, target)
            source, encoding, newline = load(target)
            old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
            if old == new:
                return "error: 'old' and 'new' are identical, nothing to do"
            if not old:
                return "error: 'old' is empty - give the exact text to replace"
            found = locate(source, old, new)
            if isinstance(found, str):
                return found
            start, end, new, note = found
            result = source[:start] + new + source[end:]
            first = source.count("\n", 0, start) + 1
            lines = snippet(result, first, first + new.count("\n"))
            broken = check(target, result)
            if broken and not check(target, source):
                return ("error: edit NOT applied - the file would no longer parse:%s\n"
                        "It would have looked like this:\n%s" % (broken, lines))
            remember(ctx, target)
            save(target, result, encoding, newline)
            return "edited %s%s%s\n%s" % (target, note, broken, lines)
        except Exception as e:
            return "error: %s" % e

    @staticmethod
    def undo(ctx, target):
        stack = (getattr(ctx, "undo", None) or {}).get(os.path.abspath(target))
        if not stack:
            return "error: no change of %s to take back in this task" % target
        data = stack.pop()
        restore(os.path.abspath(target), data)
        if data is None:
            return "undone: %s did not exist before, it is removed" % target
        return ("undone: %s is back to how it was before its last change (%d lines; %d more "
                "changes can be taken back)" % (target, data.count(b"\n"), len(stack)))


def _yes(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def locate(source, old, new):
    """(start, end, new, note) of the one place `old` stands for, or an error."""
    count = source.count(old)
    if count == 1:
        start = source.index(old)
        return start, start + len(old), new, ""
    if count > 1:
        at, lines = -1, []
        for _ in range(min(count, 10)):
            at = source.find(old, at + 1)
            lines.append(str(source.count("\n", 0, at) + 1))
        return ("error: 'old' matches %d places (lines %s). Include more of the surrounding "
                "lines so it matches exactly one." % (count, ", ".join(lines)))
    return _loose(source, old, new) or _closest(source, old)


def _indent(line):
    return len(line) - len(line.lstrip(" \t"))


def _shift(line, by, pad):
    if not line.strip():
        return line
    if by >= 0:
        return pad * by + line
    return line[min(-by, _indent(line)):]


def _loose(source, old, new):
    """Match whole lines ignoring their indentation. When exactly one place fits
    and the indentation is off by the same amount on every line, `new` is
    shifted by that amount too. The first line may differ: models often start
    `old` at the first word, not at the start of the line."""
    want = old.strip("\n").split("\n")
    stripped = [line.strip() for line in want]
    if not any(stripped):
        return None
    lines, n = source.split("\n"), len(want)
    hits = [i for i in range(len(lines) - n + 1)
            if all(lines[i + k].strip() == stripped[k] for k in range(n))]
    if len(hits) > 1:
        return ("error: 'old' matches %d places once indentation is ignored (lines %s). "
                "Include more of the surrounding lines so it matches exactly one."
                % (len(hits), ", ".join(str(i + 1) for i in hits[:10])))
    if not hits:
        return None
    i = hits[0]
    head = _indent(lines[i]) - _indent(want[0])
    rest = {_indent(lines[i + k]) - _indent(want[k]) for k in range(1, n) if stripped[k]}
    if len(rest) > 1:
        return None
    by = rest.pop() if rest else head
    # ponytail: indentation is counted in characters; a file mixing tabs and spaces gets the first line's character
    pad = lines[i][0] if lines[i][:1] in (" ", "\t") else " "
    body = new.strip("\n").split("\n")
    body = [_shift(body[0], head, pad)] + [_shift(line, by, pad) for line in body[1:]]
    start = sum(len(line) + 1 for line in lines[:i])
    end = start + sum(len(line) + 1 for line in lines[i:i + n]) - 1
    return start, end, "\n".join(body), " (matched ignoring indentation)"


def _closest(source, old):
    """'not found' plus the most similar lines in the file, numbered, so the next
    try can copy them instead of reading the file again."""
    missing = "error: 'old' not found in file"
    want = [line.strip() for line in old.strip("\n").split("\n")][:CLOSEST_MAX_LINES]
    lines = source.split("\n")
    n = min(len(want), len(lines))
    if not n:
        return missing
    # ponytail: one SequenceMatcher per window, fine for source files of a few thousand lines
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2("\n".join(want))
    best, at = 0.0, 0
    for i in range(len(lines) - n + 1):
        matcher.set_seq1("\n".join(line.strip() for line in lines[i:i + n]))
        if matcher.real_quick_ratio() > best and matcher.quick_ratio() > best:
            ratio = matcher.ratio()
            if ratio > best:
                best, at = ratio, i
    if best < 0.5:
        return missing
    return ("%s. The closest text is at lines %d-%d (%d%% similar) - copy it exactly, "
            "indentation included:\n%s" % (missing, at + 1, at + n, best * 100,
                                          numbered([line[:200] for line in lines[at:at + n]],
                                                   at + 1)))


def snippet(text, first, last):
    """Lines first..last of text with a little context, numbered like read."""
    lines = text.split("\n")
    low = max(1, first - SNIPPET_CONTEXT)
    high = min(len(lines), last + SNIPPET_CONTEXT, low + SNIPPET_MAX_LINES - 1)
    more = "\n   ..." if high < min(len(lines), last + SNIPPET_CONTEXT) else ""
    return numbered([line[:200] for line in lines[low - 1:high]], low) + more
