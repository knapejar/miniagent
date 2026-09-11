# -*- coding: utf-8 -*-
"""Hints: turn a known error message into the command that actually works.

The model keeps falling into the same traps - `gh auth switch NAME` instead of
`--user NAME`, pushing to a repository nobody created, `grep` on Windows. The
error it gets back is technically correct and practically useless, so it guesses
the next flag and burns ten steps on it.

Hints live in JSON, one file per tool, so growing the agent's competence at a
tool means adding an entry to that tool's file - no code change:

    hints/sh.json         shell, git, gh, findstr, jq
    hints/read.json       ...
    hints/browse.json     ...
    hints/any.json        applies to every tool

Each entry is {"when": regex, "say": text} and may override the tool with
"tool": "name". The regex is matched against the tool's output; when it fires,
the text is appended to that observation. MINIAGENT_HINTS points at a second
directory whose files are merged in, so your own hints live outside the repo.
"""
import json
import os
import re

HINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hints")
ANY = "any"
MAX_PER_RUN = 2          # the same hint twice is help, five times is noise


class Hints(object):
    def __init__(self, directory=None):
        self.entries = []
        self.given = {}
        for tool, path in _files(directory):
            self.entries.extend(_load(path, tool))
        # A hint written for a specific tool beats the catch-all, whatever order
        # the files happened to load in.
        self.entries.sort(key=lambda entry: entry["tool"] == ANY)

    def match(self, tool, observation):
        """Text of the first hint that applies to this tool, or an empty string."""
        if not observation:
            return ""
        for entry in self.entries:
            if entry["tool"] not in (ANY, tool):
                continue
            if not entry["_re"].search(observation):
                continue
            key = entry["say"]
            if self.given.get(key, 0) >= MAX_PER_RUN:
                continue
            self.given[key] = self.given.get(key, 0) + 1
            return "\n\n[HINT] " + entry["say"]
        return ""

    def by_tool(self):
        counts = {}
        for entry in self.entries:
            counts[entry["tool"]] = counts.get(entry["tool"], 0) + 1
        return counts

    def reset(self):
        self.given.clear()

    def __len__(self):
        return len(self.entries)


def _files(directory):
    """(tool, path) pairs. The tool comes from the file name, so adding hints
    for a tool is a matter of editing that tool's file."""
    found = []
    for base in (directory or HINTS_DIR, os.getenv("MINIAGENT_HINTS")):
        if not base or not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith(".json"):
                found.append((name[:-5], os.path.join(base, name)))
    return found


def _load(path, tool):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    out = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict) or not entry.get("say") or not entry.get("when"):
            continue
        try:
            entry["_re"] = re.compile(entry["when"], re.I | re.M)
        except re.error:
            continue
        entry.setdefault("tool", tool)
        out.append(entry)
    return out
