# -*- coding: utf-8 -*-
"""Bring LM Studio up for the local profile.

When nothing answers on the configured URL, or the model is not loaded, this
starts the LM Studio server and loads the model with the `lms` CLI. A model
that is not downloaded is replaced by the first downloaded LLM, so the agent
always has something to talk to.
"""
import json
import os
import shutil
import subprocess
import urllib.request
from urllib.parse import urlparse

from .protocol import dialect_for

LOAD_TIMEOUT = 300


def find_lms():
    path = shutil.which("lms")
    if path:
        return path
    for name in ("lms.exe", "lms"):
        candidate = os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", name)
        if os.path.isfile(candidate):
            return candidate
    return None


def served_models(cfg, timeout=3):
    """Model ids the server offers, or None when it does not answer."""
    request = urllib.request.Request(cfg.url.rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + (cfg.api_key or "none")})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [m.get("id") for m in data.get("data") or []]
    except (OSError, ValueError):
        return None


def _lms(lms, *args, timeout=60):
    return subprocess.run([lms] + list(args), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _json(lms, *args):
    try:
        return json.loads(_lms(lms, *args).stdout or "[]")
    except (ValueError, OSError, subprocess.SubprocessError):
        return []


def setup(cfg, say=print):
    """Make cfg.model answer on cfg.url. Returns True when it does."""
    host = (urlparse(cfg.url).hostname or "").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        return served_models(cfg) is not None
    lms = find_lms()
    if lms is None:
        return served_models(cfg) is not None

    try:
        if served_models(cfg) is None:
            say("LM Studio server is not running, starting it")
            _lms(lms, "server", "start", "--port", str(urlparse(cfg.url).port or 1234))
            if served_models(cfg, timeout=10) is None:
                say("LM Studio server did not start")
                return False

        loaded = [m.get("identifier") or m.get("modelKey") for m in _json(lms, "ps", "--json")]
        if cfg.model in loaded:
            return True

        downloaded = [m.get("modelKey") for m in _json(lms, "ls", "--json")
                      if m.get("type") == "llm"]
        if cfg.model not in downloaded:
            if loaded:
                choice = loaded[0]
            elif downloaded:
                choice = downloaded[0]
            else:
                say("no LLM downloaded in LM Studio")
                return False
            say("model %s is not available, using %s" % (cfg.model, choice))
            cfg.model = choice
            cfg.dialect = dialect_for(choice)
            if choice in loaded:
                return True

        say("loading %s into LM Studio (ctx %d)" % (cfg.model, cfg.ctx))
        result = _lms(lms, "load", cfg.model, "--context-length", str(cfg.ctx), "-y",
                      timeout=LOAD_TIMEOUT)
        if result.returncode != 0:
            say("lms load failed: %s" % (result.stderr or result.stdout).strip()[-300:])
            return False
        return True
    except (OSError, subprocess.SubprocessError) as e:
        say("LM Studio setup failed: %s" % e)
        return False
