# -*- coding: utf-8 -*-
"""Terminal rendering.

The Renderer is the only consumer of loop events. The loop itself prints
nothing, so the UI can be replaced or switched off without touching the logic.
"""
import sys
import threading
import time

from .ansi import (SPINNER_FRAMES, Style, bar, clear_line, human_time, human_tokens)
from .markdown import MarkdownStream

RESULT_LINES = 8          # how many lines of tool output to show
BULLET = "●"
ELBOW = "⎿"


class Spinner(object):
    """Live line while generating: frame, tokens, tokens/s, elapsed."""

    def __init__(self, style, lock, label):
        self.style, self.lock, self.label = style, lock, label
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not sys.stdout.isatty():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            with self.lock:
                frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
                sys.stdout.write(clear_line() + self.style.cyan(frame) + " " +
                                 self.style.dim(self.label()))
                sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def stop(self):
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        with self.lock:
            sys.stdout.write(clear_line())
            sys.stdout.flush()
        self._thread = None


class Renderer(object):
    def __init__(self, cfg, metrics):
        self.cfg = cfg
        self.metrics = metrics
        self.s = Style(cfg.color)
        self.lock = threading.Lock()
        self.spinner = None
        self._reset_stream()

    # ------------------------------------------------------------ streaming
    def _reset_stream(self):
        self._buf = ""
        self._mode = None          # None = undecided yet, then "text" or "tool"
        self._tokens = 0
        self._reasoning = 0
        self._t0 = time.time()
        self._md = MarkdownStream(self.s, self._emit)

    def _label(self):
        elapsed = time.time() - self._t0
        tps = self._tokens / elapsed if elapsed > 0 else 0.0
        extra = " · %s reasoning" % human_tokens(self._reasoning) if self._reasoning else ""
        return "%s tok · %.1f tok/s · %.1fs%s   (esc to stop)" % (
            human_tokens(self._tokens), tps, elapsed, extra)

    def step_begin(self, n, total):
        self._reset_stream()
        self.spinner = Spinner(self.s, self.lock, self._label)
        self.spinner.start()

    def on_delta(self, kind, piece):
        """Called by the client for every chunk of the reply."""
        self._tokens += 1
        if kind == "reasoning":
            self._reasoning += 1
            if self.cfg.show_reasoning:
                self._emit(self.s.grey(piece))
            return
        self._buf += piece
        if self._mode is None:
            head = self._buf.lstrip()
            if "<tool_call>" in head:
                self._mode = "tool"
                return
            if len(head) >= 12:
                self._mode = "text"
                self._stop_spinner()
                self._md.feed(head)
            return
        if self._mode == "text":
            if "<tool_call>" in self._buf:
                self._mode = "tool"                # the rest is a tool call
                self._emit("\n")
                self.spinner = Spinner(self.s, self.lock, self._label)
                self.spinner.start()
                return
            self._md.feed(piece)

    def _emit(self, text):
        with self.lock:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _stop_spinner(self):
        if self.spinner:
            self.spinner.stop()
            self.spinner = None

    # --------------------------------------------------------------- events
    def handle(self, event, data):
        fn = getattr(self, "_ev_" + event, None)
        if fn:
            fn(data)

    def _ev_step(self, d):
        self.step_begin(d["n"], d["max"])

    def _ev_usage(self, d):
        self._stop_spinner()

    def _ev_tool_call(self, d):
        tool = d.get("tool")
        summary = tool.summary(d["args"]) if tool else str(d["args"])[:100]
        print("%s %s%s" % (self.s.cyan(BULLET), self.s.bold(d["name"]),
                           self.s.dim("(" + summary + ")")))

    def _ev_tool_result(self, d):
        lines = d["text"].splitlines() or ["(no output)"]
        colour = self.s.red if d["error"] else self.s.grey
        for i, line in enumerate(lines[:RESULT_LINES]):
            prefix = "  %s  " % ELBOW if i == 0 else "     "
            print(prefix + colour(line[:200]))
        if len(lines) > RESULT_LINES:
            print("     " + self.s.dim("… +%d more lines" % (len(lines) - RESULT_LINES)))

    def _ev_plan(self, d):
        print("  " + self.s.magenta("plan updated") +
              self.s.dim(" (%d chars, /plan to show it)" % len(d["text"])))

    def _ev_crop(self, d):
        print("  " + self.s.yellow("✂ context cropped #%d" % d["n"]) +
              self.s.dim(" – %s chars dropped" % human_tokens(d["chars"])))

    def _ev_note(self, d):
        colour = self.s.yellow if d.get("level") == "warn" else self.s.dim
        print("  " + colour("! " + d["text"]))

    def _ev_final(self, d):
        self._stop_spinner()
        if self._mode == "text":
            self._md.close()            # flush the tail and any pending table
            print()
            return
        print()
        stream = MarkdownStream(self.s, self._emit)
        stream.feed(d["text"])
        stream.close()

    def _ev_end(self, d):
        self._stop_spinner()
        reason = {"final": self.s.green("done"),
                  "limit": self.s.yellow("step limit reached (raise --max-steps)"),
                  "stalled": self.s.red("stalled (3 replies with no tool call)"),
                  "cancelled": self.s.yellow("cancelled with esc"),
                  "asked": self.s.cyan("waiting for your answer"),
                  "error": self.s.red("error: %s" % d.get("error", ""))}.get(
                      d["reason"], d["reason"])
        print()
        print(self.s.dim("─" * 62))
        print("  %s   %s" % (reason, self.status_line()))

    # -------------------------------------------------------------- helpers
    def status_line(self, snap=None):
        m = self.metrics
        pct = m.context_pct(self.cfg.ctx)
        return self.s.dim(
            "context %s/%s %s %d%%  ·  %s tok  ·  %.1f tok/s  ·  %d steps  ·  %s" % (
                human_tokens(m.context_tokens), human_tokens(self.cfg.ctx),
                bar(pct, 8), int(pct), human_tokens(m.total_tokens),
                m.avg_tps, m.steps, human_time(m.elapsed)))

    def banner(self, session):
        s = self.s
        print()
        print("  " + s.cyan("▐▛███▜▌") + "   " + s.bold("miniagent") + s.dim(" 0.2.0"))
        print(" " + s.cyan("▝▜█████▛▘") + "  " + s.dim("an agent for local models"))
        print()
        print("  " + s.dim("model    ") + self.cfg.model + s.dim("   @ " + self.cfg.url))
        if getattr(self.cfg, "backend_note", ""):
            print("  " + s.dim("backend  " + self.cfg.backend_note))
        print("  " + s.dim("context  ") + human_tokens(self.cfg.ctx) +
              s.dim("   crop at %d%%   reasoning: %s   max steps: %d"
                    % (int(self.cfg.crop_at * 100), self.cfg.effort, self.cfg.max_steps)))
        print("  " + s.dim("cwd      ") + session.cwd)
        print("  " + s.dim("tools    ") + ", ".join(sorted(t.name for t in session.tools)) +
              s.dim("   approval: %s" % self.cfg.approve))
        print()
        print("  " + s.dim("/help for commands, esc stops the agent, /quit exits"))
        print()

    def approval(self, name, args):
        self._stop_spinner()
        print()
        print("  " + self.s.yellow("⚠ confirm") + " " + self.s.bold(name))
        for key, value in args.items():
            print("    " + self.s.dim(key + ": ") + str(value)[:300])
        try:
            return input("  " + self.s.bold("run it? [y/N] ")).strip().lower() in (
                "y", "yes", "a", "ano")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    def summary(self, snap):
        if not snap:
            return
        s = self.s
        print()
        print("  " + s.bold("run summary"))
        rows = [("steps", snap["steps"]),
                ("tokens total", human_tokens(snap["total_tokens"])),
                ("  prompt", human_tokens(snap["prompt_tokens"])),
                ("  generated", human_tokens(snap["completion_tokens"])),
                ("  reasoning", human_tokens(snap["reasoning_tokens"])),
                ("speed", "%.1f tok/s" % snap["avg_tps"]),
                ("wall time", human_time(snap["elapsed"])),
                ("  generating", human_time(snap["gen_seconds"])),
                ("  in tools", human_time(snap["tool_seconds"])),
                ("context crops", snap["crops"]),
                ("repeat loops", snap["loops"]),
                ("empty replies", snap["empty_replies"]),
                ("tools", ", ".join("%s×%d" % (k, v)
                                    for k, v in sorted(snap["tools"].items())) or "-")]
        for label, value in rows:
            print("    " + s.dim(label.ljust(20)) + str(value))
        print()
