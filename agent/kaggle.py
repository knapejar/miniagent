# -*- coding: utf-8 -*-
"""Qwen3.8-27B on a Kaggle TPU, served by kaggle-tpu-lab.

The other half lives in a separate repository (kaggle-tpu-lab): `launch.py
serve` pushes a Kaggle kernel that starts vLLM and opens public tunnels, and
`launch.py proxy` gives this machine a stable http://127.0.0.1:8080/v1 that
follows whichever tunnel is live and adds the API key itself. Pinggy tunnels
rotate every hour, so the proxy is the endpoint to use, not a tunnel URL.

This module is what `--kaggle` does before the first step:

  1. reads ~/.kaggle-tpu-lab.json, which `launch.py serve` wrote: the kernel,
     its ntfy progress topic and the API key;
  2. asks that ntfy topic whether the model is live - a plain HTTPS read that
     sends nothing to the kernel;
  3. starts `launch.py proxy` in the background when nothing listens on the
     URL yet, and stops it again on exit;
  4. reads the served model name and context limit from /v1/models;
  5. turns SearXNG off when it would collide with the proxy on port 8080.
"""
import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .secrets import register

STATE_FILE = os.getenv("KAGGLE_TPU_LAB_STATE",
                       os.path.join(os.path.expanduser("~"), ".kaggle-tpu-lab.json"))
NTFY = "https://ntfy.sh"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROXY_START_WAIT = 150   # launch.py proxy resolves the tunnel over ntfy before it binds
PROBE_TIMEOUT = 25
WAIT_POLL = 30
LIVE = ("ready", "serving", "heartbeat", "benchmark")
DEAD = ("failed", "auto-shutdown", "stopped", "hold")


# ------------------------------------------------------------------ inputs
def load_state(path=None):
    """What `launch.py serve` saved about the last launch, or {}."""
    try:
        with open(path or STATE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def find_launcher(environ=None):
    """kaggle-tpu-lab's launch.py: $KAGGLE_TPU_LAB (the directory or the file),
    else a kaggle-tpu-lab checkout next to miniagent or in the current directory."""
    environ = os.environ if environ is None else environ
    candidates = []
    if environ.get("KAGGLE_TPU_LAB"):
        candidates.append(environ["KAGGLE_TPU_LAB"])
    candidates += [os.path.join(HERE, "..", "kaggle-tpu-lab"),
                   os.path.join(os.getcwd(), "kaggle-tpu-lab")]
    for path in candidates:
        if os.path.isdir(path):
            path = os.path.join(path, "launch.py")
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def kernel_events(topic, hours=12, timeout=15):
    """Progress events the kernel published on ntfy (ntfy keeps them 12 h)."""
    url = "%s/%s/json?poll=1&since=%d" % (NTFY, topic, int(time.time()) - hours * 3600)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception:
        return None
    events = []
    for line in body.splitlines():
        try:
            entry = json.loads(line)
            if entry.get("event") == "message":
                events.append(json.loads(entry.get("message") or "{}"))
        except ValueError:
            continue
    return events


def kernel_phase(events):
    """(state, last event) where state is live | dead | starting | unknown."""
    if events is None:
        return "unknown", {}
    progress = [e for e in events if e.get("phase") not in ("cmd-ack", "cmd-result")]
    if not progress:
        return "starting", {}
    for event in reversed(progress):
        phase = event.get("phase")
        if phase in DEAD:
            return "dead", event
        if phase in LIVE:
            return "live", event
    return "starting", progress[-1]


# ----------------------------------------------------------------- network
def split_url(url):
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return host, port


def is_local(url):
    return split_url(url)[0] in ("127.0.0.1", "localhost", "::1")


def port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe(url, api_key, timeout=PROBE_TIMEOUT):
    """GET /models. Returns (info, None) with id and max_model_len, or (None, error)."""
    request = urllib.request.Request(
        url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + (api_key or "none"),
                 "ngrok-skip-browser-warning": "1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return None, "HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:200])
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return None, str(getattr(e, "reason", e))
    except ValueError:
        return None, "not an OpenAI-compatible answer (is something else on this port?)"
    models = data.get("data") or []
    if not models:
        return None, "the server lists no models"
    return {"id": models[0].get("id"), "max_model_len": models[0].get("max_model_len"),
            "ids": [m.get("id") for m in models]}, None


# ------------------------------------------------------------------- proxy
class Proxy(object):
    """`python launch.py proxy`, run by miniagent and stopped when it exits."""

    def __init__(self, launcher, host, port, log_path):
        self.launcher, self.host, self.port, self.log_path = launcher, host, port, log_path
        self.process = None

    def start(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        log = open(self.log_path, "a", encoding="utf-8")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)   # esc/ctrl-c stay ours
        self.process = subprocess.Popen(
            [sys.executable, self.launcher, "proxy", "--host", self.host,
             "--port", str(self.port)],
            cwd=os.path.dirname(self.launcher), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        log.close()
        atexit.register(self.stop)
        return self

    def wait_listening(self, timeout=PROXY_START_WAIT, tick=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                return False
            if port_open(self.host, self.port):
                return True
            if tick:
                tick()
            time.sleep(1)
        return False

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(5)
            except Exception:
                pass

    def log_tail(self, lines=8):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:]).rstrip()
        except OSError:
            return ""


# ------------------------------------------------------------------- setup
def _avoid_searxng_clash(url, environ=None):
    """miniagent looks for SearXNG on 127.0.0.1:8080 - the proxy's port. Web
    search would then send its queries to the model server."""
    environ = os.environ if environ is None else environ
    if environ.get("MINIAGENT_SEARX") is not None or not is_local(url):
        return False
    if split_url(url)[1] == 8080:
        environ["MINIAGENT_SEARX"] = "off"
        return True
    return False


_proxy = None            # the proxy this process started, if any


def setup(cfg, say=print, state_path=None):
    """Prepare cfg for the Kaggle model. Safe to call again (/kaggle).
    Returns True when the model answered."""
    global _proxy
    state = load_state(state_path)
    # A key given by the user wins; one taken from an older launch is refreshed.
    if state.get("api_key") and cfg.api_key in ("", "none", getattr(cfg, "_state_key", None)):
        cfg.api_key = cfg._state_key = state["api_key"]
    cfg.api_key = cfg.api_key or "none"
    register([cfg.api_key, state.get("api_key", "")])
    notes = ["kaggle-tpu-lab", state.get("kernel") or "no launch yet"]
    cfg.backend_note = "   ".join(notes)

    if _avoid_searxng_clash(cfg.url):
        say("web search skips SearXNG: the model's proxy holds port 8080 "
            "(MINIAGENT_SEARX=<url> points to one elsewhere)")
    if not state:
        say("no %s - start the model first: `python launch.py serve` in kaggle-tpu-lab"
            % STATE_FILE)

    # 1. is the kernel live? (read-only; sends nothing to the kernel)
    phase, event = _kernel_state(state)
    while cfg.wait and phase == "starting":
        say("waiting for the Kaggle kernel - %s (checking every %d s, ctrl-c gives up)"
            % (event.get("phase") or "queued / booting", WAIT_POLL))
        time.sleep(WAIT_POLL)
        phase, event = _kernel_state(state)
    if phase == "dead":
        say("the Kaggle kernel has ended (%s) - `python launch.py serve` starts a new one"
            % event.get("phase"))
        return False
    if phase == "starting":
        say("the Kaggle kernel is not live yet%s - `python launch.py status -f` shows it; "
            "type /kaggle here once it is, or start with --wait"
            % (" (last step: %s)" % event["phase"] if event.get("phase") else ""))
        return False
    if event.get("max_model_len"):
        _apply_limits(cfg, {"max_model_len": event["max_model_len"]})

    # 2. something to talk to: the proxy, started here when it is not running
    host, port = split_url(cfg.url)
    if not is_local(cfg.url):
        notes.append("direct tunnel URL")
    elif port_open(host, port):
        notes.append("proxy " + ("started by miniagent" if _proxy else "on :%d" % port))
    elif not cfg.spawn_proxy or not state:
        say("nothing listens on %s - run `python launch.py proxy`" % cfg.url)
    else:
        launcher = find_launcher()
        if not launcher:
            say("nothing listens on %s and kaggle-tpu-lab was not found - set "
                "KAGGLE_TPU_LAB to its directory, or run `python launch.py proxy`" % cfg.url)
        else:
            log_path = os.path.join(os.path.abspath(cfg.trace_dir), "proxy.log")
            say("starting `launch.py proxy --port %d` (log: %s)..." % (port, log_path))
            proxy = Proxy(launcher, host, port, log_path).start()
            if proxy.wait_listening():
                _proxy = proxy
                notes.append("proxy started by miniagent")
            else:
                say("the proxy did not start:\n" + proxy.log_tail())
                proxy.stop()
    cfg.backend_note = "   ".join(notes)

    # 3. the model itself: name and context limit
    while True:
        info, error = probe(cfg.url, cfg.api_key)
        if info:
            _apply_limits(cfg, info)
            return True
        if not cfg.wait:
            say("the model does not answer on %s yet: %s" % (cfg.url, error))
            return False
        say("waiting for the endpoint - %s" % error)
        time.sleep(WAIT_POLL)


def _kernel_state(state):
    if not state.get("topic"):
        return "unknown", {}
    return kernel_phase(kernel_events(state["topic"]))


def _apply_limits(cfg, info):
    if info.get("id") and cfg.model not in (info.get("ids") or [info["id"]]):
        cfg.model = info["id"]
    limit = info.get("max_model_len")
    if isinstance(limit, int) and limit > 0 and cfg.ctx > limit:
        cfg.ctx = limit
