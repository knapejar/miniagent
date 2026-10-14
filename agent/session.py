# -*- coding: utf-8 -*-
"""Conversation history, context management and the pinned plan.

LM Studio does have a context overflow policy (Truncate Middle / Rolling
Window), but it cannot be set through the OpenAI-compatible API and has been
broken since 0.4.0. So we manage the context ourselves.

History layout:

    [0] system   - system prompt with the tool specs
    [1] user     - the FIRST task of the session
    [2] user     - the PLAN (agent's pinned memory, never cropped)
    [3:]         - the run itself, cropped from the middle when it overflows

A later task is appended to the end instead of overwriting [1]. Rewriting a
message near the front would change the token prefix there, and the server's KV
cache is only reusable as an unbroken prefix from token 0 - so the whole history
behind it would be prefilled again. Appending keeps the prefix intact and a new
task costs only its own tokens. `anchor` is the index of the task being worked
on now; cropping protects it wherever it sits.
"""
import glob
import json
import os
import threading
import time

from . import paths
from .jobs import JobRunner
from .protocol import (PLAN_HEADER, build_system, parse_tool_call, parse_tool_calls,
                       wrap_tool_response)
from .secrets import collect_env, register

CHARS_PER_TOKEN = 3.4      # starting guess, calibrated against the server
CROP_MAX_FILES = 25         # files listed in the crop marker
ELIDE_TARGET = 0.6          # shrinking alone must get under this share of the crop point
ELIDE_KEEP = 300            # chars an old long observation keeps
STALE = "[stale: this read of %s was dropped - the file changed or was read again later]"
DROPPED = "\n...[%d chars dropped to save room - run the call again if you need them]"


class Session(object):
    def __init__(self, cfg, tools):
        self.cfg = cfg
        self.tools = tools
        self.cwd = os.path.abspath(cfg.workdir)
        env_secrets = collect_env()
        register(env_secrets.values())
        self.secret_names = sorted(env_secrets)
        self.dialect = getattr(cfg, "dialect", "spark")
        self.messages = [self._system()]
        self.goal = ""
        self.anchor = 1                # index of the task being worked on now
        self.has_plan = False
        self.plan_text = ""
        self.chars_per_token = CHARS_PER_TOKEN
        self.cropped = False           # flag that forces a goal reminder
        self.crops = 0
        self.last_crop = ""            # "elided" or "cut"
        self.touched = {}              # path -> {tool: count} of cropped steps
        self.undo = {}                 # path -> what it held before each change (tools/files.py)
        self.backup_dir = None         # where the task's first copies of changed files go
        # role="tool" needs a matching assistant tool_calls field, which we do
        # not send: the history carries the call as text. A server that checks
        # answers 400 "orphan_tool_message", so the native dialect starts where
        # downgrade_tool_role() would have landed anyway, one request earlier.
        self.tool_role = "user" if self.dialect == "native" else "tool"
        self.jobs = JobRunner()
        self.background_key = threading.Event()   # ctrl+b, set by ui/keys.py
        # Fixed here, not at save time: the agent's own `cd` moves self.cwd, and
        # the session must keep landing in the project it was started in.
        self.file = os.path.join(paths.sessions_dir(self.cwd),
                                 time.strftime("%Y%m%d_%H%M%S") + ".json")

    def _system(self):
        return {"role": "system", "content": build_system(
            self.tools, self.cwd, secret_names=self.secret_names, dialect=self.dialect,
            append=getattr(self.cfg, "append_system", ""),
            multi_call=getattr(self.cfg, "multi_call", 1) or 1,
            memory=getattr(self.cfg, "memory", True))}

    def set_dialect(self, dialect):
        """Another model took over (/model): its tool-call format goes into the
        system prompt, and the conversation so far carries on."""
        self.dialect = dialect
        self.messages[0] = self._system()
        if dialect == "native" and self.tool_role == "tool":
            self.downgrade_tool_role()

    # --------------------------------------------------------------- basics
    def start(self, task):
        """Record the task and pin the plan slot behind it.

        The first task of a session lands at [1]; every later one is appended,
        so the prefix in front of it - and the server's KV cache for it - stays
        valid and the new task starts without a full prefill.
        """
        self.goal = task
        self.undo = {}
        self.backup_dir = os.path.join(paths.project_dir(self.cwd), "backups",
                                       time.strftime("%Y%m%d_%H%M%S"))
        self.messages.append({"role": "user", "content": task})
        self.anchor = len(self.messages) - 1
        if not self.has_plan:
            self.messages.insert(2, {"role": "user",
                                     "content": PLAN_HEADER + (self.plan_text or "(empty)")})
            self.has_plan = True

    def reset(self, keep_plan=True):
        """Drop the conversation. The system prompt stays, the plan optionally."""
        plan = self.plan_text if keep_plan else ""
        dropped = len(self.messages) - 1
        self.messages = self.messages[:1]
        self.has_plan = False
        self.plan_text = plan
        self.goal = ""
        self.anchor = 1
        self.cropped = False
        self.touched = {}
        if getattr(self, "failures", None):
            self.failures.counts.clear()
        return dropped

    def save(self):
        """Write the conversation down so /resume can pick it up in a later run."""
        if len(self.messages) < 2:
            return
        state = {"cwd": self.cwd, "goal": self.goal, "anchor": self.anchor,
                 "has_plan": self.has_plan, "plan_text": self.plan_text,
                 "messages": self.messages}
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        tmp = self.file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        os.replace(tmp, self.file)              # a crash mid-write keeps the old copy

    def resume(self, path):
        """Continue a saved session: its conversation, plan and working directory.
        The system prompt is rebuilt, so tools and project memory are today's."""
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        if not os.path.isdir(state["cwd"]):
            raise OSError("the directory is gone: %s" % state["cwd"])
        self.reset(keep_plan=False)
        self.file = path
        self.cwd = state["cwd"]
        self.messages = state["messages"]
        self.messages[0] = self._system()
        self.goal, self.anchor = state["goal"], state["anchor"]
        self.has_plan, self.plan_text = state["has_plan"], state["plan_text"]
        self._pin_plan()
        if self.tool_role == "user":            # saved under a dialect that sent role=tool
            self.downgrade_tool_role()

    def set_goal(self, text):
        """Replace the task being worked on, in place (`/edit`)."""
        self.goal = text
        if self.anchor < len(self.messages):
            self.messages[self.anchor] = {"role": "user", "content": text}

    def set_plan(self, text):
        """The pinned copy at [2] is refreshed only when the history is rewritten
        anyway (crop, resume). Rewriting it here broke the token prefix at [2], so
        every plan update re-prefilled the whole run behind it - and until then
        the model has the new plan in front of it, in its own plan call."""
        self.plan_text = text

    def _pin_plan(self):
        if self.has_plan:
            self.messages[2] = {"role": "user", "content": PLAN_HEADER + (self.plan_text or "(empty)")}

    def add_assistant(self, text, reasoning=""):
        """Keep the reasoning next to the answer. OpenAI-compatible servers pass
        `reasoning_content` to the chat template (Qwen renders it back as <think>),
        and /history shows it."""
        if text or reasoning:
            message = {"role": "assistant", "content": text}
            if reasoning:
                message["reasoning_content"] = reasoning
            self.messages.append(message)

    def add_user(self, text):
        """An instruction from the operator's side of the conversation. An
        instruction delivered as a tool observation instead arrives wrapped in
        <tool_response>, and a small model reads that as output to comment on
        rather than as something it has been told to do."""
        self.messages.append({"role": "user", "content": text})

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

    def add_observations(self, texts):
        """The results of several calls made in one reply, in their order. As a
        user message they travel together, one <tool_response> each; as role
        "tool" each is a message of its own."""
        if len(texts) == 1 or self.tool_role == "tool":
            for text in texts:
                self.add_observation(text)
            return
        self.messages.append({"role": self.tool_role,
                              "content": "\n".join(wrap_tool_response(t) for t in texts)})

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
        return sum(_chars(m) for m in self.messages)

    def estimated_tokens(self):
        return int(self.total_chars() / self.chars_per_token)

    def calibrate(self, prompt_tokens):
        """Learn the chars-per-token ratio from the server's real numbers -
        counting content characters alone underestimates chat template overhead."""
        if prompt_tokens > 0:
            self.chars_per_token = max(1.0, float(self.total_chars()) / prompt_tokens)

    def should_crop(self):
        return self.estimated_tokens() > self.cfg.ctx * self.cfg.crop_at

    def crop(self, keep_tail=None):
        """Make room, cheapest first. Between the head (system + first task +
        plan) and the most recent messages, what is stale is shrunk in place:
        old reasoning, reads of a file that changed or was read again since,
        long tool output. The calls stay, so the run still reads as what was
        done in which order. Only when that is not enough is the middle cut.

        `keep_tail` cuts regardless, keeping that many messages: a checkpoint.
        Returns the characters freed; `last_crop` says which it was.
        """
        head = 3 if self.has_plan else 2
        tail = self.cfg.keep_tail if keep_tail is None else keep_tail
        if len(self.messages) <= head + tail + 1:
            return 0
        freed = self._elide(head, len(self.messages) - tail)
        self.last_crop = "elided"
        if keep_tail is None and freed and \
                self.estimated_tokens() <= self.cfg.ctx * self.cfg.crop_at * ELIDE_TARGET:
            self._pin_plan()
            self.cropped = True
            self.crops += 1
            return freed
        self.last_crop = "cut"
        return freed + self._cut(head, tail)

    def _elide(self, low, high):
        """Shrink, in place, what in messages[low:high] is no longer worth its room."""
        calls, several = {}, set()
        limit = getattr(self.cfg, "multi_call", 1) or 1
        for i, message in enumerate(self.messages):
            if message["role"] != "assistant":
                continue
            found = parse_tool_calls(message.get("content") or "", limit)
            if found:
                calls[i] = found[0]
            if len(found) > 1:
                several.add(i)
        freed = 0
        for i in range(low, high):
            message = self.messages[i]
            if message["role"] == "assistant":
                # a 4B model once carried 15k tokens of three failed spirals along
                freed += len(message.pop("reasoning_content", None) or "")
                continue
            call = calls.get(i - 1)
            if not call or i == self.anchor:
                continue
            text = message.get("content") or ""
            if i - 1 in several:
                # several results in one message: each is shrunk on its own
                parts = text.split("</tool_response>")
                short = "</tool_response>".join(
                    _shrink(part) for part in parts)
                if len(short) < len(text):
                    freed += len(text) - len(short)
                    message["content"] = short
                continue
            wrapped = text.startswith("<tool_response>")
            inner = text[len("<tool_response>"):-len("</tool_response>")] if wrapped else text
            path = str((call[1] or {}).get("path") or "")
            if call[0] == "read" and path and any(
                    j > i and later[0] in ("read", "edit", "write")
                    and self._short(str((later[1] or {}).get("path") or "")) == self._short(path)
                    for j, later in calls.items()):
                short = STALE % path
            elif len(inner) > ELIDE_KEEP * 2:
                short = inner[:ELIDE_KEEP] + DROPPED % (len(inner) - ELIDE_KEEP)
            else:
                continue
            if len(short) >= len(inner):
                continue
            freed += len(inner) - len(short)
            message["content"] = wrap_tool_response(short) if wrapped else short
        return freed

    def checkpoint(self):
        """A plan item just became DONE: what it took to get there is history.
        Everything behind the head goes into one marker but the plan call and its
        answer, so the next item starts in a small, clean context."""
        return self.crop(keep_tail=2)

    def _cut(self, head, tail):
        low, high = head, len(self.messages) - tail
        # A task appended after the first one sits inside the cut region; it is
        # the goal anchor, so it survives and moves just behind the marker.
        anchored = low <= self.anchor < high
        cut = self.messages[low:high]
        if anchored:
            cut = cut[:self.anchor - low] + cut[self.anchor - low + 1:]
        dropped = sum(_chars(m) for m in cut)
        self._note_files(cut)
        keep = [{"role": "user", "content": self._crop_marker(len(cut))}]
        if anchored:
            keep.append(self.messages[self.anchor])
        elif self.anchor >= high:                    # anchor still in the tail
            self.anchor -= (high - low) - len(keep)
        self.messages[low:high] = keep
        if anchored:
            self.anchor = low + 1
        self._pin_plan()
        self.cropped = True
        self.crops += 1
        return dropped

    def _note_files(self, cut):
        """Count the read/edit/write calls of the steps being cropped; an edit
        or write whose answer was an error changed nothing and is left out."""
        limit = getattr(self.cfg, "multi_call", 1) or 1
        for i, message in enumerate(cut):
            if message["role"] != "assistant":
                continue
            calls = parse_tool_calls(message.get("content") or "", limit)
            answer = (cut[i + 1].get("content") or "") if i + 1 < len(cut) else ""
            answers = answer.split("</tool_response>") if len(calls) > 1 else [answer]
            for k, call in enumerate(calls):
                if call[0] not in ("read", "edit", "write"):
                    continue
                path = str((call[1] or {}).get("path") or "")
                said = (answers[k] if k < len(answers) else "").replace(
                    "<tool_response>", "").lstrip()
                if not path or (call[0] != "read" and said.startswith("error")):
                    continue
                counts = self.touched.setdefault(self._short(path), {})
                counts[call[0]] = counts.get(call[0], 0) + 1

    def _short(self, path):
        try:
            rel = os.path.relpath(os.path.join(self.cwd, path), self.cwd)
        except ValueError:                        # another drive
            return path
        return path if rel.startswith("..") else rel

    def _crop_marker(self, steps):
        text = "[context cropped: %d earlier steps omitted]" % steps
        if not self.touched:
            return text
        rows = ["  %s: %s" % (path, ", ".join("%s %dx" % kv for kv in sorted(counts.items())))
                for path, counts in list(self.touched.items())[-CROP_MAX_FILES:]]
        return (text + "\nFiles used in the omitted steps - their content is no longer in "
                "view, read a file again before you edit it:\n" + "\n".join(rows))

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


def saved_sessions(directory=None, limit=50):
    """Saved sessions, newest first: those in `directory`, or every project's."""
    pattern = os.path.join(directory or os.path.join(paths.home(), "projects", "*",
                                                     "sessions"), "*.json")
    # ponytail: parses each file whole; keep a small header if lists get slow
    found = []
    for path in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)[:limit]:
        try:
            with open(path, encoding="utf-8") as fh:
                state = json.load(fh)
            title = state["messages"][1]["content"]
        except (OSError, ValueError, KeyError, IndexError):
            continue
        found.append({"path": path, "cwd": state.get("cwd", ""), "title": title,
                      "messages": len(state["messages"]),
                      "time": os.path.getmtime(path)})
    return found


def _shrink(part):
    """One <tool_response>...  piece of a multi-result message, cut to ELIDE_KEEP."""
    head, tag, inner = part.rpartition("<tool_response>")
    if not tag or len(inner) <= ELIDE_KEEP * 2:
        return part
    return head + tag + inner[:ELIDE_KEEP] + DROPPED % (len(inner) - ELIDE_KEEP)


def _chars(message):
    return len(message.get("content") or "") + len(message.get("reasoning_content") or "")
