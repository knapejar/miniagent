# -*- coding: utf-8 -*-
"""Configuration - one place for every default."""
import argparse
import os

from .protocol import dialect_for


class Config(object):
    """Everything that shapes a run. Built from command line arguments."""

    def __init__(self, **kw):
        self.url = kw.get("url", os.getenv("MINIAGENT_URL", "http://127.0.0.1:1234/v1"))
        self.model = kw.get("model", os.getenv("MINIAGENT_MODEL", "spark-x2.5-4b"))
        self.api_key = kw.get("api_key", os.getenv("MINIAGENT_KEY", "lmstudio"))
        # Which text format the model was trained to call tools in. Guessed from
        # the model name, because getting it wrong costs every single step.
        self.dialect = kw.get("dialect", "auto")
        if self.dialect == "auto":
            self.dialect = dialect_for(self.model)

        self.workdir = kw.get("workdir", ".")
        self.ctx = kw.get("ctx", 65536)
        self.max_steps = kw.get("max_steps", 60)
        self.max_tokens = kw.get("max_tokens", 8192)

        # Spark-X2.5 official recommendation: temperature 1.0, top_p 0.95, top_k -1.
        self.temperature = kw.get("temperature", 1.0)
        self.top_p = kw.get("top_p", 0.95)
        self.top_k = kw.get("top_k", -1)
        # LM Studio ignores chat_template_kwargs; the only working switch for
        # reasoning is reasoning_effort. On large tasks the model loops forever
        # inside <think>, so reasoning is off by default.
        self.effort = kw.get("effort", "none")

        self.crop_at = kw.get("crop_at", 0.7)      # share of ctx that triggers cropping
        self.keep_tail = kw.get("keep_tail", 8)    # messages the crop keeps at the end
        self.remind_every = kw.get("remind_every", 6)

        self.approve = kw.get("approve", "auto")   # auto | all | none
        self.stream = kw.get("stream", True)
        self.color = kw.get("color", True)
        self.show_reasoning = kw.get("show_reasoning", False)
        self.redact = kw.get("redact", True)       # mask secrets in tool output
        self.async_web = kw.get("async_web", False)  # never block on web tools
        self.trace_dir = kw.get("trace_dir", "runs")
        self.run_name = kw.get("run_name", None)
        self.timeout = kw.get("timeout", 900)

    def as_dict(self):
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and k != "api_key"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="miniagent",
        description="Agentic CLI for local models (Spark-X2.5 in LM Studio).")
    p.add_argument("task", nargs="*", help="the task; omit it for interactive mode")
    p.add_argument("-f", "--task-file", help="read the task from a file")
    p.add_argument("-C", "--workdir", default=".", help="working directory for the agent")

    g = p.add_argument_group("model")
    g.add_argument("--url", default=os.getenv("MINIAGENT_URL", "http://127.0.0.1:1234/v1"))
    g.add_argument("--model", default=os.getenv("MINIAGENT_MODEL", "spark-x2.5-4b"))
    g.add_argument("--api-key", default=os.getenv("MINIAGENT_KEY", "lmstudio"))
    g.add_argument("--effort", choices=["default", "none", "low", "medium", "high"],
                   default="none", help="reasoning_effort (default: none)")
    g.add_argument("--dialect", choices=["auto", "spark", "nanbeige", "openai"],
                   default=os.getenv("MINIAGENT_DIALECT", "auto"),
                   help="tool-call format: spark | nanbeige text formats, or openai = "
                        "standard function calling (default: guessed from the model name)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-tokens", type=int, default=8192)
    g.add_argument("--ctx", type=int, default=65536)

    g = p.add_argument_group("loop")
    g.add_argument("--max-steps", type=int, default=60)
    g.add_argument("--crop-at", type=float, default=0.7)
    g.add_argument("--keep-tail", type=int, default=8)
    g.add_argument("--remind-every", type=int, default=6)
    g.add_argument("--approve", choices=["auto", "all", "none"], default="auto")

    g = p.add_argument_group("output")
    g.add_argument("--no-stream", action="store_true", help="disable streaming")
    g.add_argument("--no-color", action="store_true")
    g.add_argument("--show-reasoning", action="store_true", help="print the model's reasoning")
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
    cfg = Config(url=a.url, model=a.model, api_key=a.api_key, workdir=a.workdir,
                 ctx=a.ctx, max_steps=a.max_steps, max_tokens=a.max_tokens,
                 temperature=a.temperature, top_p=a.top_p, effort=a.effort,
                 dialect=a.dialect,
                 crop_at=a.crop_at, keep_tail=a.keep_tail, remind_every=a.remind_every,
                 approve=a.approve, stream=not a.no_stream, color=not a.no_color,
                 show_reasoning=a.show_reasoning, redact=not a.no_redact,
                 async_web=a.async_web, run_name=a.run_name)
    return cfg, task
