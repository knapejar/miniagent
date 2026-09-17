# -*- coding: utf-8 -*-
"""Conversation history, context management and the pinned plan.

LM Studio does have a context overflow policy (Truncate Middle / Rolling
Window), but it cannot be set through the OpenAI-compatible API and has been
broken since 0.4.0. So we manage the context ourselves.

History layout:

    [0] system   - system prompt with the tool specs
    [1] user     - the ORIGINAL task (goal anchor, cropping never touches it)
    [2] user     - the PLAN (agent's pinned memory, never cropped)
    [3:]         - the run itself, cropped from the middle when it overflows
"""
import os
import threading

from .jobs import JobRunner
from .protocol import PLAN_HEADER, build_system, wrap_tool_response
from .secrets import collect_env, register

CHARS_PER_TOKEN = 3.4      # starting guess, calibrated against the server


class Session(object):
    def __init__(self, cfg, tools):
        self.cfg = cfg
        self.tools = tools
        self.cwd = os.path.abspath(cfg.workdir)
        env_secrets = collect_env()
        register(env_secrets.values())
        self.secret_names = sorted(env_secrets)
        self.dialect = getattr(cfg, "dialect", "spark")
        self.messages = [{"role": "system",
                          "content": build_system(tools, self.cwd,
                                                  secret_names=self.secret_names,
                                                  dialect=self.dialect)}]
        self.goal = ""
        self.has_plan = False
        self.plan_text = ""
        self.chars_per_token = CHARS_PER_TOKEN
        self.cropped = False           # flag that forces a goal reminder
        self.crops = 0
        self.tool_role = "tool"
        self.jobs = JobRunner()
        self.background_key = threading.Event()   # ctrl+b, set by ui/keys.py

    # --------------------------------------------------------------- basics
    def start(self, task):
        """Record the task and pin the plan slot behind it."""
        self.goal = task
        self.messages.append({"role": "user", "content": task})
        if not self.has_plan:
            self.messages.insert(2, {"role": "user",
                                     "content": PLAN_HEADER + (self.plan_text or "(empty)")})
            self.has_plan = True
        else:
            # The anchor at [1] is what cropping protects; it has to be the task
            # being worked on now, not the first one of the session.
            self.messages[1] = {"role": "user", "content": task}

    def reset(self, keep_plan=True):
        """Drop the conversation. The system prompt stays, the plan optionally."""
        plan = self.plan_text if keep_plan else ""
        dropped = len(self.messages) - 1
        self.messages = self.messages[:1]
        self.has_plan = False
        self.plan_text = plan
        self.goal = ""
        self.cropped = False
        if getattr(self, "failures", None):
            self.failures.counts.clear()
        return dropped

    def set_plan(self, text):
        self.plan_text = text
        if self.has_plan:
            self.messages[2] = {"role": "user", "content": PLAN_HEADER + text}

    def add_assistant(self, text):
        if text:
            self.messages.append({"role": "assistant", "content": text})

    def add_observation(self, text):
        # Qwen's chat template wraps a tool message in <tool_response> itself;
        # wrapping it here too would nest the tags.
        if self.tool_role == "tool" and self.dialect == "qwen":
            content = text
        else:
            content = wrap_tool_response(text)
        message = {"role": self.tool_role, "content": content}
        if self.tool_role == "tool":
            message["tool_call_id"] = "c%d" % len(self.messages)
        self.messages.append(message)

    def downgrade_tool_role(self):
        """Some servers reject role='tool' without a structured tool_calls field."""
        self.tool_role = "user"
        for message in self.messages:
            if message["role"] == "tool":
                message["role"] = "user"
                message.pop("tool_call_id", None)
                if not message["content"].startswith("<tool_response>"):
                    message["content"] = wrap_tool_response(message["content"])

    # -------------------------------------------------------------- context
    def total_chars(self):
        return sum(len(m.get("content") or "") for m in self.messages)

    def estimated_tokens(self):
        return int(self.total_chars() / self.chars_per_token)

    def calibrate(self, prompt_tokens):
        """Learn the chars-per-token ratio from the server's real numbers -
        counting content characters alone underestimates chat template overhead."""
        if prompt_tokens > 0:
            self.chars_per_token = max(1.0, float(self.total_chars()) / prompt_tokens)

    def should_crop(self):
        return self.estimated_tokens() > self.cfg.ctx * self.cfg.crop_at

    def crop(self):
        """Truncate the middle. The head (system + task + plan) and the most
        recent messages stay; everything between them becomes one marker.

        TODO: summarise instead of dropping, and never summarise the sections
        currently being worked on (last opened file / running command).
        """
        head = 3 if self.has_plan else 2
        tail = self.cfg.keep_tail
        if len(self.messages) <= head + tail + 1:
            return 0
        cut = self.messages[head:-tail]
        dropped = sum(len(m.get("content") or "") for m in cut)
        self.messages[head:-tail] = [{
            "role": "user",
            "content": "[context cropped: %d earlier steps omitted]" % len(cut)}]
        self.cropped = True
        self.crops += 1
        return dropped

    # ------------------------------------------------------------- reminder
    def reminder(self, step, max_steps):
        """Goal anchor appended to the end of an observation - the last thing the
        model reads before its next turn. Returns the text or an empty string."""
        due = self.cfg.remind_every and step % self.cfg.remind_every == 0
        if not (self.cropped or due):
            return ""
        self.cropped = False
        goal = self.goal if len(self.goal) <= 600 else self.goal[:600] + "..."
        return ("\n\n[GOAL] %s\n[step %d/%d - keep going until the goal is fully done "
                "and verified. Update the plan if anything changed status.]"
                % (goal, step, max_steps))
