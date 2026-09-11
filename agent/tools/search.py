# -*- coding: utf-8 -*-
"""Regex search across files.

Added because the model kept reaching for a `grep` tool that did not exist and
burned a dozen steps on `findstr` invocations with grep syntax that Windows
does not accept. Giving it the tool it already expects is cheaper than teaching
it findstr.
"""
import os
import re

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}
MAX_MATCHES = 60
MAX_LINE = 200
BINARY_HINT = b"\x00"


class GrepTool(object):
    name = "grep"
    description = ("Search file contents by regex, returns path:line: text. "
                   "glob limits filenames (*.py), path limits the subtree.")
    params = {"pattern": {"type": "string"},
              "path": {"type": "string", "description": "file or directory, default: working directory"},
              "glob": {"type": "string", "description": "filename filter such as *.py"}}
    required = ("pattern",)
    danger = False

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return False

    def summary(self, args):
        extra = ""
        if args.get("glob"):
            extra += " in %s" % args["glob"]
        if args.get("path"):
            extra += " under %s" % args["path"]
        return "%s%s" % (str(args.get("pattern", ""))[:60], extra)

    def run(self, ctx, pattern=None, path=None, glob=None, **_):
        if not pattern:
            return "error: missing arg 'pattern'"
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return "error: bad regular expression (%s)" % e

        root = path if path and os.path.isabs(path) else os.path.join(ctx.cwd, path or "")
        if not os.path.exists(root):
            return "error: no such file or directory: %s" % root

        hits, scanned, truncated = [], 0, False
        for filename in self._walk(root, glob):
            scanned += 1
            try:
                with open(filename, "rb") as fh:
                    blob = fh.read()
                if BINARY_HINT in blob[:2048]:
                    continue
                text = blob.decode("utf-8", "replace")
            except Exception:
                continue
            rel = os.path.relpath(filename, ctx.cwd)
            for number, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append("%s:%d: %s" % (rel, number, line.strip()[:MAX_LINE]))
                    if len(hits) >= MAX_MATCHES:
                        truncated = True
                        break
            if truncated:
                break

        if not hits:
            return "no matches (%d files searched)" % scanned
        tail = "\n...[stopped at %d matches]" % MAX_MATCHES if truncated else ""
        return "\n".join(hits) + tail

    @staticmethod
    def _walk(root, glob):
        import fnmatch
        if os.path.isfile(root):
            yield root
            return
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                yield os.path.join(base, name)
