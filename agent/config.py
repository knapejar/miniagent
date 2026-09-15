# -*- coding: utf-8 -*-
"""Configuration - one place for every default."""
import argparse
import os

from .protocol import DIALECTS, dialect_for

EFFORTS = ("default", "none", "low", "medium", "high", "xhigh")

# A profile is a set of defaults for one kind of backend. Anything given on the
# command line (or to Config directly) wins over the profile.
PROFILES = {
    # A model on this machine, e.g. Spark-X2.5-4B in LM Studio.
    "local": {
        "url": lambda: os.getenv("MINIAGENT_URL", "http://127.0.0.1:1234/v1"),
        "model": lambda: os.getenv("MINIAGENT_MODEL", "spark-x2.5-4b"),
        "api_key": lambda: os.getenv("MINIAGENT_KEY", "lmstudio"),
        "ctx": 65536,
        "max_tokens": 8192,
        # Spark-X2.5 official recommendation: temperature 1.0, top_p 0.95, top_k -1.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        # LM Studio ignores chat_template_kwargs; the only working switch for
        # reasoning is reasoning_effort. On large tasks the model loops forever
        # inside <think>, so reasoning is off by default.
        "effort": "none",
        "retries": 0,
        "outage_wait": 0,
    },
    # Qwen3.8-27B on a Kaggle TPU, started by kaggle-tpu-lab (`launch.py serve`)
    # and reached through its local proxy (`launch.py proxy`). See agent/kaggle.py.
    "kaggle": {
        "url": lambda: os.getenv("MINIAGENT_KAGGLE_URL", "http://127.0.0.1:8080/v1"),
        "model": lambda: os.getenv("MINIAGENT_KAGGLE_MODEL", "qwen3.8-27b"),
        # The proxy puts the real key in itself; a direct tunnel URL needs the
        # key from ~/.kaggle-tpu-lab.json, which agent/kaggle.py fills in.
        "api_key": lambda: os.getenv("MINIAGENT_KAGGLE_KEY", ""),
        # The server takes 262k, but vllm-tpu 0.28 has no prefix caching for
        # this model, so every step re-reads the whole prompt: 131k keeps a step
        # under ~10 s. The real limit from /v1/models caps it further.
        "ctx": 131072,
        "max_tokens": 16384,
        # None = not sent, so vLLM applies the model's own generation_config.
        "temperature": None,
        "top_p": None,
        "top_k": None,
        # A 27B keeps its reasoning on track; xhigh is slow per step, medium is
        # the balance for an agent. /effort changes it live.
        "effort": "medium",
        # Tunnels rotate and the proxy answers 502 while it re-resolves one.
        "retries": 3,
        # A crashed vLLM engine restarts in place in ~15-20 min; the agent waits for
        # it instead of dying and keeps the conversation.
        "outage_wait": 1800,
    },
}


class Config(object):
    """Everything that shapes a run. Built from command line arguments."""

    def __init__(self, **kw):
        self.profile = kw.get("profile") or "local"
        base = PROFILES[self.profile]

        def pick(name):
            if kw.get(name) is not None:
                return kw[name]
            value = base[name]
            return value() if callable(value) else value

        self.url = pick("url")
        self.model = pick("model")
        self.api_key = pick("api_key")
        dialect = kw.get("dialect") or "auto"
        if dialect == "auto":
            dialect = "qwen" if self.profile == "kaggle" else dialect_for(self.model)
        self.dialect = dialect                      # tool-call format, see protocol.py

        self.workdir = kw.get("workdir", ".")
        self.ctx = pick("ctx")
        self.max_steps = kw.get("max_steps", 60)
        self.max_tokens = pick("max_tokens")

        self.temperature = pick("temperature")      # None = not sent to the server
        self.top_p = pick("top_p")
        self.top_k = pick("top_k")
        self.effort = pick("effort")
        self.retries = pick("retries")
        self.outage_wait = pick("outage_wait")     # seconds to wait out a server outage

        self.crop_at = kw.get("crop_at", 0.7)      # share of ctx that triggers cropping
        self.keep_tail = kw.get("keep_tail", 8)    # messages the crop keeps at the end
        self.remind_every = kw.get("remind_every", 6)

        self.approve = kw.get("approve", "auto")   # auto | all | none
        self.stream = kw.get("stream", True)
        self.color = kw.get("color", True)
        self.show_reasoning = kw.get("show_reasoning", True)
        self.redact = kw.get("redact", True)       # mask secrets in tool output
        self.async_web = kw.get("async_web", False)  # never block on web tools
        self.trace_dir = kw.get("trace_dir", "runs")
        self.run_name = kw.get("run_name", None)
        self.timeout = kw.get("timeout", 900)

        # kaggle profile only
        self.spawn_proxy = kw.get("spawn_proxy", True)   # start launch.py proxy if needed
        self.wait = kw.get("wait", False)                # wait until the model answers
        self.backend_note = ""                           # one banner line, set at startup

    def as_dict(self):
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and k != "api_key"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="miniagent",
        description="Agentic CLI for local models (Spark-X2.5 in LM Studio), or for "
                    "Qwen3.8-27B on a Kaggle TPU with --kaggle.")
    p.add_argument("task", nargs="*", help="the task; omit it for interactive mode")
    p.add_argument("-f", "--task-file", help="read the task from a file")
    p.add_argument("-C", "--workdir", default=".", help="working directory for the agent")

    g = p.add_argument_group("model")
    g.add_argument("--kaggle", action="store_true",
                   help="Qwen3.8-27B served by kaggle-tpu-lab (same as --profile kaggle)")
    g.add_argument("--profile", choices=sorted(PROFILES),
                   default=os.getenv("MINIAGENT_PROFILE", "local"),
                   help="set of defaults for a backend (default: local)")
    g.add_argument("--url", help="OpenAI-compatible endpoint")
    g.add_argument("--model", help="model identifier")
    g.add_argument("--api-key")
    g.add_argument("--dialect", choices=("auto",) + DIALECTS, default="auto",
                   help="tool-call format: spark | qwen (default: from the model name)")
    g.add_argument("--effort", choices=EFFORTS,
                   help="reasoning effort (local: none, kaggle: medium)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--ctx", type=int)

    g = p.add_argument_group("kaggle")
    g.add_argument("--no-proxy", action="store_true",
                   help="do not start `launch.py proxy` when nothing answers on the URL")
    g.add_argument("--wait", action="store_true",
                   help="wait until the Kaggle model is live before starting")

    g = p.add_argument_group("loop")
    g.add_argument("--max-steps", type=int, default=60)
    g.add_argument("--crop-at", type=float, default=0.7)
    g.add_argument("--keep-tail", type=int, default=8)
    g.add_argument("--remind-every", type=int, default=6)
    g.add_argument("--approve", choices=["auto", "all", "none"], default="auto")

    g = p.add_argument_group("output")
    g.add_argument("--no-stream", action="store_true", help="disable streaming")
    g.add_argument("--no-color", action="store_true")
    g.add_argument("--no-reasoning", action="store_true",
                   help="do not print the model's reasoning (shown live by default)")
    g.add_argument("--no-redact", action="store_true", help="do not mask secrets in output")
    g.add_argument("--async-web", action="store_true",
                   help="websearch and browse never block; results arrive later")
    g.add_argument("--run-name", help="name of the trace file")
    p.add_argument("-v", "--version", action="version", version="miniagent 0.2.0")

    a = p.parse_args(argv)
    task = " ".join(a.task)
    if not task and a.task_file:
        with open(a.task_file, encoding="utf-8") as fh:
            task = fh.read().strip()
    given = {name: getattr(a, name) for name in
             ("url", "model", "api_key", "effort", "ctx", "max_tokens",
              "temperature", "top_p", "top_k") if getattr(a, name) is not None}
    cfg = Config(profile="kaggle" if a.kaggle else a.profile, dialect=a.dialect,
                 workdir=a.workdir, max_steps=a.max_steps,
                 crop_at=a.crop_at, keep_tail=a.keep_tail, remind_every=a.remind_every,
                 approve=a.approve, stream=not a.no_stream, color=not a.no_color,
                 show_reasoning=not a.no_reasoning, redact=not a.no_redact,
                 async_web=a.async_web, run_name=a.run_name,
                 spawn_proxy=not a.no_proxy, wait=a.wait, **given)
    return cfg, task
