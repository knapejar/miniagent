# -*- coding: utf-8 -*-
"""Where miniagent keeps its own state: one home directory, not a dot-folder
scattered into every project it ever touches.

Everything lives under ~/.miniagent/projects/<slug>/, where the slug is the
project's absolute path with its separators flattened to dashes - the same
convention Claude Code uses, so the folder name still reads as a path:

    C:\\Users\\Jarda\\REPO\\Moje\\miniagent
        -> ~/.miniagent/projects/C--Users-Jarda-REPO-Moje-miniagent/
               memory/INDEX.md      notes the agent writes for its next run
               runs/*.jsonl         one trace per run

Both halves sit side by side on purpose: the agent can grep its own history of
this project the same way it greps anything else.
"""
import os
import re

ENV_HOME = "MINIAGENT_HOME"


def home():
    """The root of miniagent's state, ~/.miniagent unless MINIAGENT_HOME says otherwise."""
    return os.path.abspath(os.environ.get(ENV_HOME)
                           or os.path.join(os.path.expanduser("~"), ".miniagent"))


def slug(cwd="."):
    """A project path flattened into one filename-safe component."""
    path = os.path.abspath(cwd)
    if os.name == "nt":
        path = path.replace("/", "\\")
    return re.sub(r"[^A-Za-z0-9]", "-", path).rstrip("-") or "root"


def project_dir(cwd="."):
    return os.path.join(home(), "projects", slug(cwd))


def memory_dir(cwd="."):
    return os.path.join(project_dir(cwd), "memory")


def runs_dir(cwd="."):
    return os.path.join(project_dir(cwd), "runs")


def memory_index(cwd="."):
    return os.path.join(memory_dir(cwd), "INDEX.md")
