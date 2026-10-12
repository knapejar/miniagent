# -*- coding: utf-8 -*-
"""What the agent is shown of the project before its first step, and of its
own changes before it calls a task done.

The 4B model once spent a whole 30-step run searching the disk for the file
it was asked to edit - the file sat in the working directory, which it never
listed. A few lines of orientation in the system prompt cost less than one step.
"""
import json
import os
import subprocess

LISTING_MAX = 40          # top-level entries shown
CHILDREN_MAX = 6          # names shown inside each top-level directory
STATUS_MAX = 10           # lines of git status


def git(cwd, *args):
    try:
        done = subprocess.run(["git", "-C", cwd] + list(args), capture_output=True,
                              timeout=5, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.decode("utf-8", "replace").rstrip() if done.returncode == 0 else ""


def _entries(path):
    from .tools.search import SKIP_DIRS       # here: tools import protocol, which imports this
    try:
        names = sorted(os.listdir(path), key=lambda n: (not os.path.isdir(os.path.join(path, n)),
                                                        n.lower()))
    except OSError:
        return []
    return [n for n in names if n not in SKIP_DIRS and not n.startswith(".")]


def listing(cwd):
    rows = []
    entries = _entries(cwd)
    for name in entries[:LISTING_MAX]:
        path = os.path.join(cwd, name)
        if os.path.isdir(path):
            inner = _entries(path)
            shown = ", ".join(inner[:CHILDREN_MAX]) + (", ..." if len(inner) > CHILDREN_MAX else "")
            rows.append("  %s/  (%d: %s)" % (name, len(inner), shown) if inner else "  %s/  (empty)" % name)
        else:
            rows.append("  " + name)
    if len(entries) > LISTING_MAX:
        rows.append("  ... %d more" % (len(entries) - LISTING_MAX))
    return rows


def test_command(cwd):
    """How this project runs its tests, when that can be told from its files."""
    def has(*parts):
        return os.path.exists(os.path.join(cwd, *parts))

    def text(name):
        try:
            with open(os.path.join(cwd, name), encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    if has("package.json"):
        try:
            if json.loads(text("package.json")).get("scripts", {}).get("test"):
                return "npm test"
        except (ValueError, AttributeError):
            pass
    if has("Cargo.toml"):
        return "cargo test"
    if has("go.mod"):
        return "go test ./..."
    pytest = has("pytest.ini") or has("conftest.py") or "[tool.pytest" in text("pyproject.toml")
    for folder in ("tests", "test"):
        if has(folder) and any(n.endswith(".py") and (n.startswith("test") or n.endswith("_test.py"))
                               for n in os.listdir(os.path.join(cwd, folder))):
            return "python -m pytest -q" if pytest else "python -m unittest discover -s %s" % folder
    return "python -m pytest -q" if pytest else ""


def snapshot(cwd):
    """A few lines about the working directory, for the system prompt."""
    if not os.path.isdir(cwd):
        return ""
    rows = listing(cwd)
    if not rows:
        return "\n\nThe working directory is empty."
    out = ["", "", "Working directory at the start of this session:"] + rows
    branch = git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        status = git(cwd, "status", "--short").splitlines()
        out.append("git: branch %s, %s" % (branch, "%d changed files" % len(status) if status else "clean"))
        out.extend("  " + line for line in status[:STATUS_MAX])
        out.extend("  " + line[:100] for line in git(cwd, "log", "--oneline", "-3").splitlines())
    tests = test_command(cwd)
    if tests:
        out.append("Tests run with: " + tests)
    return "\n".join(out)


def diff_stat(cwd):
    """What git sees changed, or "" outside a repository."""
    return git(cwd, "diff", "--stat")
