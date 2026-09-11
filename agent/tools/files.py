# -*- coding: utf-8 -*-
"""File access: read / write / edit.

`edit` is deliberately the simplest possible patch - an exact string
replacement. For a small model that is more reliable than a unified-diff
envelope, and the failure message ("'old' not found") is trivial to act on.
"""
import json
import os

READ_DEFAULT_LINES = 200


def validate(path):
    """Cheap syntax check so a broken write is caught now, not ten steps later."""
    try:
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
    except Exception:
        return ""
    if path.endswith(".py"):
        try:
            compile(source, path, "exec")
        except SyntaxError as e:
            return "\nSYNTAX ERROR line %s: %s" % (e.lineno, e.msg)
    elif path.endswith(".json"):
        try:
            json.loads(source)
        except ValueError as e:
            return "\nJSON ERROR: %s" % e
    return ""


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
    description = "Read a text file with line numbers."
    params = {"path": {"type": "string"},
              "start": {"type": "integer", "description": "1-based first line, default 1"},
              "count": {"type": "integer", "description": "number of lines, default 200"}}
    required = ("path",)

    def run(self, ctx, path=None, start=1, count=READ_DEFAULT_LINES, **_):
        try:
            with open(self.resolve(ctx, path), encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except Exception as e:
            return "error: %s" % e
        first, wanted = max(1, int(start)), int(count)
        chunk = lines[first - 1:first - 1 + wanted]
        body = "\n".join("%5d| %s" % (first + i, line) for i, line in enumerate(chunk))
        more = "" if first - 1 + wanted >= len(lines) else \
            "\n...[file has %d lines]" % len(lines)
        return (body or "(empty range)") + more


class WriteTool(_FileTool):
    name = "write"
    description = "Create or overwrite a text file."
    params = {"path": {"type": "string"}, "text": {"type": "string"}}
    required = ("path", "text")

    def summary(self, args):
        text = str(args.get("text", ""))
        return "%s  (%d bytes)" % (args.get("path", "?"), len(text.encode("utf-8")))

    def run(self, ctx, path=None, text="", **_):
        try:
            target = self.resolve(ctx, path)
            if not text and os.path.exists(target) and os.path.getsize(target) > 0:
                return ("error: refusing to overwrite %s with empty content. If you meant "
                        "to clear it, delete it with sh first." % target)
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            return "wrote %d bytes to %s%s" % (
                len(text.encode("utf-8")), target, validate(target))
        except Exception as e:
            return "error: %s" % e


class EditTool(_FileTool):
    name = "edit"
    description = "Replace the first exact occurrence of old with new in a file."
    params = {"path": {"type": "string"}, "old": {"type": "string"},
              "new": {"type": "string"}}
    required = ("path", "old", "new")

    def summary(self, args):
        # Showing truncated old/new makes distinct edits look identical, which
        # hides thrashing. Show sizes and the first differing fragment instead.
        old, new = str(args.get("old", "")), str(args.get("new", ""))
        head = old.strip().splitlines()[0][:40] if old.strip() else "(empty)"
        return "%s  %r  %d->%d chars" % (args.get("path", "?"), head, len(old), len(new))

    def run(self, ctx, path=None, old="", new="", **_):
        try:
            target = self.resolve(ctx, path)
            with open(target, encoding="utf-8") as fh:
                source = fh.read()
            if old == new:
                return "error: 'old' and 'new' are identical, nothing to do"
            if old not in source:
                return "error: 'old' not found in file"
            with open(target, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(source.replace(old, new, 1))
            return "edited %s%s" % (target, validate(target))
        except Exception as e:
            return "error: %s" % e
