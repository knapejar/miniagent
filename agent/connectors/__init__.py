# -*- coding: utf-8 -*-
"""Connectors: what makes a profile's backend answer before the first step.

A profile names one with "connector" (default lmstudio). Each module has
    setup(cfg, say)  bring the backend up - start a server, load the model;
                     True once the model answers
    status(cfg)      (ready, note) without starting anything, for /model
"""
import importlib
import threading

from ..jobs import parallel

NAMES = ("lmstudio", "llamacpp", "kaggle")


def get(name):
    if name not in NAMES:
        raise ValueError("unknown connector %r (one of %s)" % (name, ", ".join(NAMES)))
    return importlib.import_module("." + name, __name__)


def connect(cfg, say=print):
    """Run the profile's connector; sets and returns cfg.ready."""
    try:
        cfg.ready = bool(get(cfg.connector).setup(cfg, say=say))
    except ValueError as e:
        say(str(e))
        cfg.ready = False
    return cfg.ready


def status(cfg):
    """(ready, note) for cfg's profile: whether it answers now, and if not,
    what picking it would do."""
    try:
        return get(cfg.connector).status(cfg)
    except Exception as e:                  # one broken profile must not hide the rest
        return False, "error: %s" % e


def overview():
    """[(provider, name, model, url, ready, note)] for every profile, sorted
    by provider; the servers are asked in parallel."""
    cfgs = _configs()
    return _rows(cfgs, parallel(cfgs, status))


def overview_live():
    """overview() without the wait, for the /model picker: (rows, busy).
    rows() is overview() so far - a server not answered yet says "asking..." -
    and busy() is true while one is still out. The order does not change."""
    cfgs = _configs()
    states = [None] * len(cfgs)

    def ask(i):
        states[i] = status(cfgs[i])
    for i in range(len(cfgs)):
        # daemon: a server that hangs to its timeout must not hold up quitting
        threading.Thread(target=ask, args=(i,), daemon=True).start()
    return (lambda: _rows(cfgs, [s or (False, "asking...") for s in states]),
            lambda: None in states)


def _configs():
    from ..config import Config, load_profiles
    return [Config(profile=name) for name in load_profiles()]


def _rows(cfgs, states):
    rows = [(c.provider, c.profile, c.model, c.url) + s for c, s in zip(cfgs, states)]
    return sorted(rows, key=lambda r: (r[0] != "lmstudio", r[0], r[1]))


def row_text(row, current=None):
    provider, name, model, url, ready, note = row
    host = url.split("//")[-1].split("/")[0]
    return "%-9s %-28s %-24s %-22s %s%s" % (
        provider, name, model, host, "ready" if ready else note,
        "   <- current" if name == current else "")


def profiles_text(current=None):
    """What `--profile` without a name prints."""
    from .. import paths
    from ..config import stored_default
    current = current or stored_default()
    lines = ["  profiles (--profile <name>, --set-default remembers it; /model inside)"]
    lines += ["    " + row_text(row, current) for row in overview()]
    lines.append("  yours live in %s, one JSON file per provider" % paths.profiles_dir())
    return "\n".join(lines)
