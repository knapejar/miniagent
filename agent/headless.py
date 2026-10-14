# -*- coding: utf-8 -*-
"""Headless mode: miniagent driven by another program instead of a person.

    miniagent -p "task"                                   answer on stdout
    miniagent -p "task" --output-format json              one result object
    miniagent -p --input-format stream-json --output-format stream-json \\
              --include-partial-messages --session-id <uuid>

The flags and the stream are those of Claude Code's print mode, so whatever
drives `claude -p` drives miniagent the same way. With stream-json input every
stdin line is one message:

    {"type": "user", "message": {"role": "user", "content": "text or blocks"}}
    {"type": "control_request", "request_id": "r1", "request": {"subtype": "interrupt"}}

and every stdout line one event: system/init, stream_event (text and thinking
deltas, tool_use starts), assistant (text and tool_use blocks), user
(tool_result blocks) and result at the end of each task. The process keeps
the conversation between messages and saves it under its session id, so
`--resume <id>` continues it in a later process.
"""
import glob
import json
import os
import queue
import signal
import sys
import threading
import time
import uuid

from . import mcp, paths
from .llm import LLMClient
from .loop import AgentLoop
from .protocol import THINK_RE
from .session import Session
from .tools import default_tools
from .tools.shell import kill_all
from .trace import Trace

CALL_TAGS = ("<tool_call>", "<think>")


class Output(object):
    """Lines on stdout, one JSON object each, from any thread."""

    def __init__(self, fmt):
        self.fmt = fmt
        self.lock = threading.Lock()

    def emit(self, obj):
        if self.fmt != "stream-json":
            return
        line = json.dumps(obj, ensure_ascii=False)
        with self.lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def raw(self, text):
        with self.lock:
            sys.stdout.write(text)
            sys.stdout.flush()


def say(text):
    """Diagnostics go to stderr; stdout belongs to the protocol."""
    try:
        sys.stderr.write("miniagent: %s\n" % text)
        sys.stderr.flush()
    except Exception:
        pass


def visible(text):
    """The part of a reply streamed so far that is prose for the reader: no
    reasoning, nothing from the tool call on, and no half-written tag at the end."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = THINK_RE.sub("", text)
    for tag in CALL_TAGS:
        if tag in text:
            text = text.split(tag, 1)[0]
    for tag in CALL_TAGS:
        for k in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:k]):
                return text[:-k]
    return text


def text_of(content):
    """A user message's text. Images cannot reach a text-only endpoint; the
    model is told one was there."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif block.get("type") == "image":
            parts.append("[an image was attached here; this model cannot see images]")
    return "\n".join(p for p in parts if p)


def find_session(session_id, cwd):
    name = "%s.json" % session_id
    here = os.path.join(paths.sessions_dir(cwd), name)
    if os.path.exists(here):
        return here
    found = glob.glob(os.path.join(paths.home(), "projects", "*", "sessions", name))
    return found[0] if found else None


class Step(object):
    """One model call as Claude Code streams it: a message whose text and
    thinking blocks grow by deltas."""

    def __init__(self, out, partial):
        self.out, self.partial = out, partial
        self.raw, self.sent, self.block, self.index, self.open = "", 0, None, -1, False

    def start(self):
        self.finish()
        self.raw, self.sent, self.block, self.index = "", 0, None, -1
        self.open = True
        self._event({"type": "message_start", "message": {
            "id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
            "content": []}})

    def _event(self, event):
        if self.partial:
            self.out.emit({"type": "stream_event", "event": event})

    def _switch(self, kind):
        if self.block == kind:
            return
        if self.block is not None:
            self._event({"type": "content_block_stop", "index": self.index})
        self.block, self.index = kind, self.index + 1
        empty = {"type": "text", "text": ""} if kind == "text" else \
            {"type": "thinking", "thinking": ""}
        self._event({"type": "content_block_start", "index": self.index, "content_block": empty})

    def on_delta(self, kind, text):
        if kind == "reasoning" and text:
            self._switch("thinking")
            self._event({"type": "content_block_delta", "index": self.index,
                         "delta": {"type": "thinking_delta", "thinking": text}})
        elif kind == "content" and text:
            self.raw += text
            shown = visible(self.raw)
            if len(shown) > self.sent:
                self._switch("text")
                self._event({"type": "content_block_delta", "index": self.index,
                             "delta": {"type": "text_delta", "text": shown[self.sent:]}})
                self.sent = len(shown)

    def tool_start(self, tool_id, name):
        if self.block is not None:
            self._event({"type": "content_block_stop", "index": self.index})
        self.block, self.index = None, self.index + 1
        self._event({"type": "content_block_start", "index": self.index, "content_block": {
            "type": "tool_use", "id": tool_id, "name": name, "input": {}}})
        self._event({"type": "content_block_stop", "index": self.index})

    def finish(self):
        if not self.open:
            return
        if self.block is not None:
            self._event({"type": "content_block_stop", "index": self.index})
        self._event({"type": "message_stop"})
        self.open, self.block = False, None


class Headless(object):
    def __init__(self, cfg, opts):
        self.cfg, self.opts = cfg, opts
        self.out = Output(opts["output_format"])
        self.inbox = queue.Queue()
        self.cancel = None                  # the running task's event
        self.memory_cancel = threading.Event()
        self.tools = default_tools()
        self.mcp_status, self.mcp_handles = [], []
        self.num_turns = 0
        self.connected = False

    # ------------------------------------------------------------- setup
    def setup(self):
        servers = mcp.load_config(self.opts["mcp_config"])
        if servers:
            found, self.mcp_status, self.mcp_handles = mcp.start(servers, say=say)
            self.tools += found
        banned = set(self.opts["disallowed"])
        self.tools = [t for t in self.tools if t.name not in banned]
        self.session = Session(self.cfg, self.tools)
        self.client = LLMClient(self.cfg, self.tools)
        resume = self.opts["resume"]
        if resume:
            path = find_session(resume, self.session.cwd)
            if not path:
                say("No conversation found with session ID: %s" % resume)
                return False
            cwd = self.session.cwd
            self.session.resume(path)
            if os.path.isdir(cwd):          # where we were started wins over where it was saved
                self.session.cwd = cwd
            self.session_id = resume
        else:
            self.session_id = self.opts["session_id"] or str(uuid.uuid4())
            self.session.file = os.path.join(paths.sessions_dir(self.session.cwd),
                                             "%s.json" % self.session_id)
        self.trace = Trace(self.cfg.trace_dir, self.cfg.run_name or self.session_id)
        return True

    def connect(self):
        """Bring the backend up once, before the first task needs it."""
        if self.connected:
            return
        from . import connectors
        self.connected = True
        if not connectors.connect(self.cfg, say=say):
            say("%s (%s @ %s) is not ready - trying anyway" % (
                self.cfg.profile, self.cfg.model, self.cfg.url))
        self.session.set_dialect(self.cfg.dialect)

    def init_event(self):
        self.out.emit({
            "type": "system", "subtype": "init", "session_id": self.session_id,
            "cwd": self.session.cwd, "model": self.cfg.model, "profile": self.cfg.profile,
            "tools": [t.name for t in self.tools], "mcp_servers": self.mcp_status,
            "permissionMode": "bypassPermissions" if self.cfg.approve == "none" else "default",
            "uuid": str(uuid.uuid4())})

    # ------------------------------------------------------------- input
    def reader(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                say("not JSON, ignored: %s" % line[:80])
                continue
            if message.get("type") == "control_request":
                self.control(message)
            elif message.get("type") == "user":
                self.memory_cancel.set()    # a new task cuts the memory pass short
                self.inbox.put(message)
        self.inbox.put(None)

    def control(self, message):
        request = message.get("request") or {}
        kind = request.get("subtype")
        response = {"subtype": "success", "request_id": message.get("request_id")}
        if kind == "interrupt":
            if self.cancel is not None:
                self.cancel.set()
            self.memory_cancel.set()
        elif kind == "get_context_usage":
            used, ctx = self.session.estimated_tokens(), self.cfg.ctx
            response["response"] = {"totalTokens": used, "maxTokens": ctx,
                                    "percentage": round(100.0 * used / ctx, 1) if ctx else None,
                                    "cropAt": self.cfg.crop_at}
        else:
            response = {"subtype": "error", "request_id": message.get("request_id"),
                        "error": "unsupported control request: %s" % kind}
        self.out.emit({"type": "control_response", "response": response})

    # -------------------------------------------------------------- tasks
    def task(self, text):
        started = time.time()
        self.num_turns += 1
        self.init_event()
        if text.strip() == "/compact":
            return self.compact(started)
        self.connect()
        cancel = self.cancel = threading.Event()
        loop = AgentLoop(self.cfg, self.session, self.client, self.tools, self.trace,
                         approve_fn=None, cancel=cancel)
        step = Step(self.out, self.opts["partial"])
        final, reason, error, memory_due = "", "final", None, False
        tokens = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        tool_id = None
        try:
            for event, data in loop.run(text, on_delta=step.on_delta):
                if event == "step":
                    step.start()
                    # Saved as it goes, not only at the end: a process killed
                    # mid-task (a stop that did not land, a restart) still leaves a
                    # conversation that --resume finds.
                    self.save()
                elif event == "usage":
                    usage = data["usage"]
                    tokens["input_tokens"] += usage.prompt_tokens or 0
                    tokens["output_tokens"] += usage.completion_tokens or 0
                    tokens["cache_read_input_tokens"] += getattr(usage, "cached_tokens", 0) or 0
                elif event == "tool_call":
                    tool_id = "toolu_" + uuid.uuid4().hex[:24]
                    if step.open:
                        step.tool_start(tool_id, data["name"])
                        step.finish()
                    # several calls of one reply: the prose goes with the first
                    prose = data.get("prose", "")
                    blocks = [{"type": "text", "text": prose}] if prose else []
                    blocks.append({"type": "tool_use", "id": tool_id, "name": data["name"],
                                   "input": data["args"]})
                    self.assistant(blocks)
                elif event == "tool_result":
                    self.out.emit({"type": "user", "session_id": self.session_id,
                                   "uuid": str(uuid.uuid4()), "message": {
                                       "role": "user", "content": [{
                                           "type": "tool_result", "tool_use_id": tool_id,
                                           "content": data["text"],
                                           "is_error": bool(data.get("error"))}]}})
                elif event == "final":
                    step.finish()
                    final = data["text"]
                    if final.strip():
                        self.assistant([{"type": "text", "text": final}])
                elif event == "crop":
                    self.out.emit({"type": "system", "subtype": "compact_boundary",
                                   "session_id": self.session_id,
                                   "compact_metadata": {"trigger": "auto",
                                                        "chars": data.get("chars")}})
                elif event == "note":
                    self.out.emit({"type": "system", "subtype": "notice",
                                   "session_id": self.session_id,
                                   "level": data.get("level"), "text": data.get("text")})
                elif event == "end":
                    reason, error = data.get("reason"), data.get("error")
                    memory_due = bool(data.get("memory_due")) and not cancel.is_set()
        except Exception as e:                 # a failed task must not end the process
            reason, error = "error", "%s: %s" % (type(e).__name__, e)
            self.trace.write("crash", err=error)
        finally:
            step.finish()
            self.cancel = None
        self.save()
        is_error = reason == "error"
        self.result(started, final if not is_error else (error or "error"), is_error,
                    reason, tokens, loop.metrics.steps)
        if memory_due:
            self.memory(loop)

    def save(self):
        try:
            self.session.save()
        except OSError as e:
            say("session not saved: %s" % e)

    def assistant(self, blocks):
        self.out.emit({"type": "assistant", "session_id": self.session_id,
                       "uuid": str(uuid.uuid4()), "parent_tool_use_id": None,
                       "message": {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message",
                                   "role": "assistant", "model": self.cfg.model,
                                   "content": blocks}})

    def result(self, started, text, is_error, reason, tokens, steps):
        ms = int((time.time() - started) * 1000)
        self.last = {"type": "result",
                     "subtype": "error_during_execution" if is_error else "success",
                     "is_error": is_error, "stop_reason": reason, "result": text,
                     "duration_ms": ms, "duration_api_ms": ms, "num_turns": steps,
                     "session_id": self.session_id, "total_cost_usd": 0, "usage": tokens,
                     "uuid": str(uuid.uuid4())}
        self.out.emit(self.last)

    def compact(self, started):
        before = self.session.estimated_tokens()
        freed = self.session.crop(keep_tail=4)
        after = self.session.estimated_tokens()
        if freed:
            self.out.emit({"type": "system", "subtype": "compact_boundary",
                           "session_id": self.session_id,
                           "compact_metadata": {"trigger": "manual", "pre_tokens": before,
                                                "post_tokens": after}})
        try:
            self.session.save()
        except OSError:
            pass
        self.result(started, "compacted" if freed else "nothing to compact", False, "final",
                    {"input_tokens": 0, "output_tokens": 0}, 0)

    def memory(self, loop):
        """The memory pass, after the result: a new message cuts it short."""
        self.memory_cancel.clear()
        if not self.inbox.empty():
            return
        try:
            for event, data in loop.memory_pass(self.memory_cancel):
                if event == "memory_save" and data.get("wrote"):
                    self.out.emit({"type": "system", "subtype": "notice",
                                   "session_id": self.session_id, "level": "info",
                                   "text": "memory: " + ", ".join(data["wrote"])})
        except Exception as e:
            say("memory pass failed: %s" % e)

    # ---------------------------------------------------------------- run
    def interrupt_signal(self, *_):
        if self.cancel is not None:
            self.cancel.set()
        self.memory_cancel.set()

    def run(self, prompt):
        if not self.setup():
            return 1
        try:
            signal.signal(signal.SIGINT, self.interrupt_signal)
        except ValueError:                  # not the main thread (tests)
            pass
        try:
            if self.opts["input_format"] == "stream-json":
                threading.Thread(target=self.reader, daemon=True).start()
                if prompt:
                    self.task(prompt)
                while True:
                    message = self.inbox.get()
                    if message is None:
                        break
                    content = (message.get("message") or {}).get("content")
                    self.task(text_of(content))
            else:
                if not prompt:
                    prompt = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
                if not prompt:
                    say("no prompt: give one as an argument, on stdin, or use "
                        "--input-format stream-json")
                    return 1
                self.task(prompt)
                fmt = self.opts["output_format"]
                if fmt == "json":
                    self.out.raw(json.dumps(self.last, ensure_ascii=False) + "\n")
                elif fmt == "text":
                    self.out.raw((self.last.get("result") or "") + "\n")
                return 1 if self.last.get("is_error") else 0
            return 0
        finally:
            kill_all(self.session)
            for server in self.mcp_handles:
                server.close()
            self.trace.close()


def main(cfg, prompt):
    return Headless(cfg, cfg.headless).run(prompt)
