# -*- coding: utf-8 -*-
"""Bring llama-server up for a model LM Studio cannot load.

Such a model often needs its own llama.cpp build (a fork, a patched
architecture). The profile says which, with "connector": "llamacpp" and

    "server":      path to that build's llama-server
    "gguf":        path to the model
    "server_args": extra arguments, e.g. ["-ctk", "q8_0", "-fa", "on"]

When nothing answers on cfg.url, this starts that server on the model and
waits until it serves. It stays running after miniagent exits, so the next run
starts at once.
"""
import os
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

from .lmstudio import served_models

LOAD_TIMEOUT = 300


def _files(cfg):
    return [os.path.expanduser(cfg.options.get(key) or "") for key in ("server", "gguf")]


def status(cfg):
    if served_models(cfg) is not None:
        return True, "ready"
    missing = [path for path in _files(cfg) if not os.path.isfile(path)]
    if missing:
        return False, "missing %s" % (missing[0] or "server/gguf in the profile")
    return False, "starts llama-server on pick"


def setup(cfg, say=print):
    """Make cfg.model answer on cfg.url. Returns True when it does."""
    if served_models(cfg) is not None:
        return True
    server, gguf = _files(cfg)
    for path in (server, gguf):
        if not os.path.isfile(path):
            say("%s: %s not found" % (cfg.profile, path or "no server/gguf in the profile"))
            return False

    log_path = os.path.join(os.path.dirname(server), "llama-server.log")
    say("starting llama-server with %s (ctx %d), log in %s"
        % (os.path.basename(gguf), cfg.ctx, log_path))
    with open(log_path, "w", encoding="utf-8") as log:
        # Detached, so the loaded model outlives this run and the next one starts at once.
        proc = subprocess.Popen(
            [server, "-m", gguf, "-ngl", "99", "-c", str(cfg.ctx), "--parallel", "1",
             *cfg.options.get("server_args", []),
             "--host", urlparse(cfg.url).hostname or "127.0.0.1",
             "--port", str(urlparse(cfg.url).port or 8080), "-a", cfg.model],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0))

    deadline = time.time() + LOAD_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                say("llama-server exited: %s" % fh.read().strip()[-400:])
            return False
        # /models answers while the model is still loading; /health turns 200 once it is in.
        if served_models(cfg) is not None and _healthy(cfg):
            return True
        time.sleep(1)
    say("llama-server did not come up in %ds" % LOAD_TIMEOUT)
    return False


def _healthy(cfg):
    base = cfg.url.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False
