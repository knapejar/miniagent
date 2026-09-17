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

    def elapsed_time(self):
        """Returns the time the current run has been going on, in seconds."""
        return getattr(self, "_run_start", None)

    def estimated_tokens(self):
        """Estimates total tokens in the conversation based on character count."""
        total_chars = self.total_chars()
        return int(total_chars / self.chars_per_token)

    def total_chars(self):
        """Returns the total number of characters in all messages."""
        return sum(len(str(msg.get("content", ""))) for msg in self.messages)

    def calibrate(self, tokens):
        """Calibrates the chars_per_token based on actual server token count.

        Updates chars_per_token to better match the server's token estimation
        so estimated_tokens() becomes more accurate.
        """
        if not self.messages:
            return
        # Only calibrate if we have actual token data
        if tokens and self.total_chars():
            # Adjust chars_per_token: target = total_chars / actual_tokens
            self.chars_per_token = self.total_chars() / tokens
            self.chars_per_token = max(self.chars_per_token, 1.0)

    def should_crop(self):
        return self.estimated_tokens() > self.cfg.ctx * self.cfg.crop_at

    def crop(self):
        """Truncate the middle. The head (system + task + plan) and the most
        recent messages stay; everything between them becomes one marker.

        Returns the dropped text, or an empty string when nothing was dropped.
        """
        if not self.should_crop():
            return ""
        # Build the boundary: head = system+task+plan, tail = last messages
        head_end = 3
        total = len(self.messages)
        # Keep at least head + tail; the tail is everything after head_end
        tail_start = total - 1
        head = self.messages[:head_end]
        tail = self.messages[tail_start:]
        middle = self.messages[head_end:tail_start]
        if not middle:
            return ""
        cut = middle[0]
        dropped = cut.get("content") or ""
        self.messages = head + tail
        self.cropped = True
        self.crops += 1
        return dropped

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

    def add_observation(self, text):
        # Qwen's chat template wraps a tool message in
        # itself; wrapping it here too would nest the tags.
        if self.tool_role == "tool" and self.dialect == "qwen":
            content = text
        else:
            content = wrap_tool_response(text)
        message = {"role": self.tool_role, "content": content}
        if self.tool_role == "tool":
            message["tool_call_id"] = "c%d" % len(self.messages)
        self.messages.append(message)
