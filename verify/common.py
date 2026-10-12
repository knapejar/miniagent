# -*- coding: utf-8 -*-
"""What every checker shares: import the agent's module from the workspace,
run checks that may hang or raise, print `SCORE: passed/total` last.

    python verify/h04_verify.py <workspace>
"""
import ast
import importlib
import os
import subprocess
import sys
import threading

WS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, WS)
os.chdir(WS)
sys.setrecursionlimit(5000)
if hasattr(sys.stdout, "reconfigure"):          # a cp1252 console cannot print every label
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

RESULTS = []
TIMEOUT = 5           # seconds per check; a backtracking engine may never return


class Raises(object):
    """Expected outcome: the call raises this exception (or a subclass)."""

    def __init__(self, error):
        self.error = error


def load(name):
    try:
        return importlib.import_module(name)
    except BaseException as e:                   # noqa: BLE001 - any failure is a 0
        print("FATAL: cannot import %s.py from %s: %s: %s" % (name, WS, type(e).__name__, e))
        print("SCORE: 0/1")
        sys.exit(2)


def source(name):
    try:
        with open(os.path.join(WS, name), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def forbids(group, filename, modules=(), calls=()):
    """The file imports none of `modules` and calls none of `calls` - read from
    its syntax tree, so a comment that mentions eval() is not a call."""
    try:
        tree = ast.parse(source(filename))
    except SyntaxError as e:
        RESULTS.append((group, "%s parses" % filename, str(e), False))
        return
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in modules]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in modules:
            found.append(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in calls:
            found.append(node.func.id + "()")
    RESULTS.append((group, "no %s in %s" % ("/".join(tuple(modules) + tuple(calls)), filename),
                    "found %s" % ", ".join(found), not found))


def check(group, label, fn, expect=True):
    """fn() must return expect (== comparison; a float compares to 1e-6), or
    raise when expect is Raises(...)."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:               # noqa: BLE001
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(TIMEOUT)
    if worker.is_alive():
        RESULTS.append((group, label, "timed out after %ds" % TIMEOUT, False))
        return
    if isinstance(expect, Raises):
        error = box.get("error")
        ok = isinstance(error, expect.error)
        got = "raised %r" % error if "error" in box else "returned %r" % box.get("value")
    elif "error" in box:
        ok, got = False, "raised %s: %s" % (type(box["error"]).__name__, box["error"])
    else:
        value = box.get("value")
        ok = _same(value, expect)
        got = "returned %r" % (value,)
    RESULTS.append((group, label, got[:200], ok))


def _same(value, expect):
    if isinstance(expect, float) and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        return abs(value - expect) < 1e-6
    try:
        return value == expect
    except Exception:                            # noqa: BLE001
        return False


def suite_passes(group="suite"):
    """The repository's own test suite still passes in the workspace."""
    done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                          cwd=WS, capture_output=True, timeout=240)
    tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
    RESULTS.append((group, "python -m unittest discover -s tests", " | ".join(tail),
                    done.returncode == 0))


def report():
    groups = {}
    for group, _label, _got, ok in RESULTS:
        groups.setdefault(group, [0, 0])
        groups[group][0] += ok
        groups[group][1] += 1
    for group, (passed, total) in groups.items():
        print("  %-12s %2d/%2d%s" % (group, passed, total, "" if passed == total else "  <--"))
    fails = [r for r in RESULTS if not r[3]]
    for group, label, got, _ok in fails[:20]:
        print("FAIL [%s] %s -> %s" % (group, label, got))
    passed = sum(1 for r in RESULTS if r[3])
    print("SCORE: %d/%d" % (passed, len(RESULTS)))
    sys.exit(0 if not fails else 1)
