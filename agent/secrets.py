# -*- coding: utf-8 -*-
"""Mask credentials before they reach the transcript.

Written after a run where the agent ran `gh auth token`, printed two live
GitHub OAuth tokens into the terminal, then pasted them into curl commands.
Secrets that a tool prints end up in the conversation, in the JSONL trace, and
in the model's context - from where they can be echoed again at any time.

The model never needs to see a credential to use one: it can reference an
environment variable instead.
"""
import re

PATTERNS = [
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "GitHub PAT"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "API key"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"), "Slack token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS key id"),
    (re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}"), "Google token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    # user:password@host in a URL
    (re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+(@)"), "URL credentials"),
]


SECRET_NAME = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|_key)$")
_VALUES = []


def collect_env(environ=None):
    """Names of environment variables that hold credentials. The model is told
    the names so it can reference them; the values stay out of its context."""
    import os
    environ = os.environ if environ is None else environ
    found = {}
    for name, value in environ.items():
        if SECRET_NAME.search(name) and len(value or "") >= 8:
            found[name] = value
    return found


def register(values):
    for value in values:
        if value and len(value) >= 8 and value not in _VALUES:
            _VALUES.append(value)


def redact(text, enabled=True):
    """Replace anything that looks like a credential with a labelled placeholder."""
    if not enabled or not text:
        return text
    for value in _VALUES:
        text = text.replace(value, "[REDACTED secret]")
    for pattern, label in PATTERNS:
        if label == "URL credentials":
            text = pattern.sub(r"\1[REDACTED %s]\2" % label, text)
        else:
            text = pattern.sub("[REDACTED %s]" % label, text)
    return text


def has_secret(text):
    return any(p.search(text or "") for p, _ in PATTERNS)
