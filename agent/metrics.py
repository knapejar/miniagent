# -*- coding: utf-8 -*-
"""Run metrics - what the status line and the summary show."""
import time


class Metrics(object):
    def __init__(self):
        self.started = time.time()
        self.steps = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.gen_seconds = 0.0        # time spent generating (tools excluded)
        self.tool_seconds = 0.0
        self.tool_counts = {}
        self.crops = 0
        self.empty_replies = 0
        self.loops = 0
        self.last_tps = 0.0
        self.last_ttft = 0.0
        self.context_tokens = 0

    # ----------------------------------------------------------- writing
    def add_usage(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.gen_seconds += usage.elapsed
        self.context_tokens = usage.prompt_tokens
        self.last_tps = usage.tps
        self.last_ttft = usage.ttft

    def add_tool(self, name, seconds):
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        self.tool_seconds += seconds

    # ----------------------------------------------------------- reading
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed(self):
        return time.time() - self.started

    @property
    def avg_tps(self):
        return self.completion_tokens / self.gen_seconds if self.gen_seconds > 0 else 0.0

    def context_pct(self, ctx):
        return 100.0 * self.context_tokens / ctx if ctx else 0.0

    def snapshot(self):
        return {"steps": self.steps, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
                "elapsed": round(self.elapsed, 1),
                "gen_seconds": round(self.gen_seconds, 1),
                "tool_seconds": round(self.tool_seconds, 1),
                "avg_tps": round(self.avg_tps, 1),
                "context_tokens": self.context_tokens,
                "tools": dict(self.tool_counts), "crops": self.crops,
                "empty_replies": self.empty_replies, "loops": self.loops}
