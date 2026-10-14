# -*- coding: utf-8 -*-
"""Client for an OpenAI-compatible endpoint (LM Studio, llama-server, vLLM...).

Streams the reply chunk by chunk so tokens/s can be shown live, and so a cancel
request stops generation mid-token instead of after the whole reply.
No dependencies beyond the standard library.
"""
import http.client
import json
import queue
import threading
import time
import urllib.error
import urllib.request

from .protocol import cut_after_call, format_tool_call

# Statuses a tunnel or a proxy answers with while the route to
# the server is being re-established - worth another try, unlike a 400. 429 is
# a hosted backend saying "at capacity, retry shortly", which is the same thing.
RETRY_STATUS = (429, 502, 503, 504, 520, 522, 524, 530)
RETRY_WAIT = 10
CALL_END = "</tool_call>"
OVERRUN_CHARS = 80     # text after a finished call before the stream is abandoned
THINK_CAP = 0.6        # share of max_tokens the reasoning may take before it is cut
# Qwen3.8 thinking levels, passed through its chat template (vLLM). miniagent's
# "high" is the model's top level.
QWEN_EFFORT = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}


class Cancelled(Exception):
    """esc was pressed while waiting for the server."""


class LLMError(RuntimeError):
    """Server-side failure. `overflow` means the prompt exceeded the context."""

    def __init__(self, message, status=None, transient=False):
        RuntimeError.__init__(self, message)
        self.status = status
        self.transient = transient      # the server is down or restarting; worth waiting
        low = message.lower()
        self.overflow = (("context" in low and "exceed" in low) or "too long" in low
                         or "maximum context length" in low)   # vLLM's wording


class Usage(object):
    def __init__(self, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                 finish_reason=None, ttft=0.0, elapsed=0.0, reasoning="", cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason
        self.ttft = ttft              # time to first token
        self.elapsed = elapsed
        self.reasoning = reasoning    # the reasoning text itself, kept for the history
        self.cached_tokens = cached_tokens   # prompt tokens the server served from its cache

    @property
    def tps(self):
        return self.completion_tokens / self.elapsed if self.elapsed > 0 else 0.0


def cached_of(usage):
    """Prompt tokens served from the server's prefix cache: OpenAI's
    prompt_tokens_details.cached_tokens, or DeepSeek's prompt_cache_hit_tokens."""
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0


def calls_to_text(calls, dialect="qwen", limit=1):
    """vLLM started with --enable-auto-tool-choice parses the model's tool call
    out of the answer even when the request lists no tools, and sends it as
    structured `tool_calls` with the text removed. Turn up to `limit` calls back
    into the markup protocol.py reads. Returns (text, complete): arguments that
    are not valid JSON mean the call was cut off, so it is left unclosed and
    nothing after it is kept."""
    if not calls:
        return "", True
    ordered = [calls[k] for k in sorted(calls)] if isinstance(calls, dict) else list(calls)
    texts = []
    for call in ordered[:max(1, limit)]:
        text, complete = _call_to_text(call, dialect)
        texts.append(text)
        if not complete:
            return "\n".join(texts), False
    return "\n".join(texts), True


def _call_to_text(call, dialect):
    function = call.get("function") or call
    name = (function.get("name") or "").strip()
    raw = function.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(args, dict):
            raise ValueError("arguments are not an object")
    except ValueError:
        return "<tool_call>\n<function=%s>\n" % name, False
    args = {key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for key, value in args.items()}
    return format_tool_call(name, args, dialect), True


class LLMClient(object):
    def __init__(self, cfg, tools=()):
        self.cfg = cfg
        self.tools = list(tools)      # only sent in the "native" dialect
        self.tool_role = "tool"       # probed at runtime: does the server accept it?

    @property
    def multi(self):
        """How many tool calls one reply may carry (config multi_call)."""
        return max(1, int(getattr(self.cfg, "multi_call", 1) or 1))

    # ---------------------------------------------------------- request body
    @property
    def _dialect(self):
        return getattr(self.cfg, "dialect", "spark") or "spark"

    @property
    def server_stop(self):
        """Only Spark stops on </tool_call> at the server. Qwen thinks first, and
        a server-side stop would also fire on a </tool_call> it merely writes
        while reasoning about the format - so for Qwen the client stops, on the
        answer only. In the native dialect the server ends the call itself."""
        return self._dialect == "spark"

    @property
    def think_cap_chars(self):
        """How much reasoning one reply may hold before the stream is cut. A model
        that ignores reasoning_effort can otherwise spend the whole token budget
        inside <think> and return nothing at all - three times in a row."""
        share = getattr(self.cfg, "think_cap", THINK_CAP)
        return int(self.cfg.max_tokens * 4 * share) if share else 0

    def _body(self, messages, stream, effort=None):
        effort = effort or self.cfg.effort
        body = {"model": self.cfg.model, "messages": messages, "stream": stream,
                "max_tokens": self.cfg.max_tokens}
        for name in ("temperature", "top_p", "top_k"):
            if getattr(self.cfg, name) is not None:
                body[name] = getattr(self.cfg, name)
        if self.server_stop:
            body["stop"] = [CALL_END]
        if self._dialect == "native" and self.tools:
            body["tools"] = [{"type": "function", "function": t.spec()} for t in self.tools]
        if stream:
            body["stream_options"] = {"include_usage": True}
        if getattr(self.cfg, "effort_in_template", False):
            # vLLM validates the top-level reasoning_effort against the OpenAI
            # values and Qwen3.8 reads its level from the chat template instead.
            if effort == "none":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            elif effort in QWEN_EFFORT:
                body["chat_template_kwargs"] = {"reasoning_effort": QWEN_EFFORT[effort]}
        elif effort != "default":
            # LM Studio ignores chat_template_kwargs (measured) and honours only
            # reasoning_effort; kwargs stay for other backends.
            body["reasoning_effort"] = effort
            if effort == "none":
                body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _open(self, request, cancel=None):
        """urlopen blocks until the server answers - a dead backend can hold it
        for the whole timeout, and esc has to work meanwhile. Open in a thread
        and let it clean the socket up if the wait was abandoned."""
        if cancel is None:
            return urllib.request.urlopen(request, timeout=self.cfg.timeout)
        box = {}
        def run():
            try:
                box["r"] = urllib.request.urlopen(request, timeout=self.cfg.timeout)
            except BaseException as e:            # noqa: BLE001 - re-raised below
                box["e"] = e
            if cancel.is_set() and "r" in box:
                box.pop("r").close()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while worker.is_alive() and not cancel.is_set():
            worker.join(0.1)
        if "r" in box:
            return box["r"]
        if "e" in box:
            raise box["e"]
        raise Cancelled()

    @staticmethod
    def _lines(response, cancel):
        """The response line by line, each read on request by a thread of its own,
        which also closes it. Once the link is dead (wifi gone) a read blocks until
        the timeout, and on Windows neither close() nor shutdown() from another
        thread wakes it (measured) - waiting here instead lets esc and ctrl+c
        through. Nothing is read ahead: a reply given up on is hung up on."""
        wants, lines = queue.Queue(), queue.Queue()

        def pump():
            try:
                with response:
                    rows = iter(response)
                    while wants.get():          # False: the reader is done
                        lines.put(next(rows, None))
            except BaseException as e:          # noqa: BLE001 - raised below, in the reader
                lines.put(e)
        threading.Thread(target=pump, daemon=True).start()
        try:
            wants.put(True)
            while True:
                try:
                    raw = lines.get(timeout=0.1)
                except queue.Empty:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    continue
                if raw is None:
                    return
                if isinstance(raw, BaseException):
                    raise raw
                yield raw
                wants.put(True)
        finally:
            wants.put(False)

    def _request(self, messages, stream, cancel=None, effort=None, on_delta=None):
        """Open the request. A connection that fails is tried again cfg.retries
        times; on_delta("retry", (error, seconds)) hears of each wait."""
        data = json.dumps(self._body(messages, stream, effort)).encode("utf-8")
        retries = getattr(self.cfg, "retries", 0) or 0
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                self.cfg.url.rstrip("/") + "/chat/completions", data=data, method="POST",
                headers=dict(getattr(self.cfg, "headers", None) or {},
                             **{"Content-Type": "application/json",
                                "Authorization": "Bearer " + (self.cfg.api_key or "none")}))
            try:
                return self._open(request, cancel)
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:500]
                # vLLM answers 500 "EngineCore encountered an issue" once its engine died;
                # it is restarted, so that is an outage like 502/503, not a bad request.
                transient = e.code in RETRY_STATUS or (e.code == 500 and "Engine" in text)
                error = LLMError("HTTP %s: %s" % (e.code, text), status=e.code, transient=transient)
                if not transient:
                    raise error
            except urllib.error.URLError as e:
                error = LLMError("cannot reach %s (%s)" % (self.cfg.url, e.reason), transient=True)
            except (OSError, http.client.HTTPException) as e:   # reset, timeout, dropped
                error = LLMError("connection to %s failed (%s: %s)" % (
                    self.cfg.url, type(e).__name__, e), transient=True)
            if attempt == retries:
                raise error
            wait = RETRY_WAIT * (attempt + 1)
            if on_delta:
                on_delta("retry", ("%s - retry %d/%d" % (error, attempt + 1, retries), wait))
            if cancel is not None and cancel.wait(wait):
                raise Cancelled()
            if cancel is None:
                time.sleep(wait)

    # ---------------------------------------------------------------- calls
    def chat(self, messages, on_delta=None, cancel=None, effort=None):
        """Return (text, Usage).

        `on_delta(kind, text)` receives chunks as they arrive, where kind is
        "content" or "reasoning" - or "retry" with (error, seconds) before a
        wait to try again. `cancel` is a threading.Event; when it is set the
        stream is abandoned and whatever arrived so far is returned.
        `effort` overrides the configured reasoning effort for this one call.
        """
        if self.cfg.stream:
            return self._chat_stream(messages, on_delta, cancel, effort)
        return self._chat_once(messages, on_delta, effort)

    def _chat_once(self, messages, on_delta, effort=None):
        started = time.time()
        with self._request(messages, False, effort=effort) as response:
            data = json.loads(response.read().decode("utf-8"))
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        thought = message.get("reasoning_content") or message.get("reasoning") or ""
        finish = choice.get("finish_reason")
        if message.get("tool_calls") and "<tool_call>" not in text:
            call_text, complete = calls_to_text(message["tool_calls"], self._dialect, self.multi)
            text = (text.rstrip() + "\n" if text.strip() else "") + call_text
            if not complete:
                finish = "length"
        if not self.server_stop:
            text = cut_after_call(text, self.multi)
        if on_delta and thought:
            on_delta("reasoning", thought)
        if on_delta and text:
            on_delta("content", text)
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        elapsed = time.time() - started
        return self._close(text, finish, self.multi), Usage(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            details.get("reasoning_tokens", 0), finish, elapsed, elapsed, thought,
            cached_of(usage))

    def _chat_stream(self, messages, on_delta, cancel=None, effort=None):
        started = time.time()
        ttft = 0.0
        parts, thoughts, reasoning_chars, chunks = [], [], 0, 0
        prompt_tokens = completion_tokens = reasoning_tokens = cached_tokens = 0
        ends = 0               # calls closed so far in `content`
        finish = None
        calls = {}             # index -> {"name", "arguments"} from structured tool_calls
        content = ""           # the answer so far, kept only for the client-side stop
        call_end = -1          # where </tool_call> ended in `content`
        cancelled = lambda: cancel is not None and cancel.is_set()
        try:
            cap = self.think_cap_chars if effort != "none" else 0
            response = self._request(messages, True, cancel, effort, on_delta)
            for raw in self._lines(response, cancel):
                if cancel is not None and cancel.is_set():
                    finish = "cancelled"
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(event.get("error"), dict):
                    # vLLM reports a failure that happens after the 200 inside the stream
                    message = str(event["error"].get("message", event["error"]))[:500]
                    raise LLMError("server error mid-reply: %s" % message,
                                   status=event["error"].get("code"),
                                   transient="Engine" in message)
                if event.get("usage"):
                    usage = event["usage"]
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens", reasoning_tokens)
                    cached_tokens = cached_of(usage) or cached_tokens
                for choice in event.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    for call in delta.get("tool_calls") or []:
                        entry = calls.setdefault(call.get("index", 0),
                                                 {"name": "", "arguments": ""})
                        function = call.get("function") or {}
                        entry["name"] = entry["name"] or function.get("name") or ""
                        entry["arguments"] += function.get("arguments") or ""
                        chunks += 1
                        if ttft == 0.0:
                            ttft = time.time() - started
                    if thought:
                        reasoning_chars += len(thought)
                        thoughts.append(thought)
                        chunks += 1
                        if ttft == 0.0:
                            ttft = time.time() - started
                        if on_delta:
                            on_delta("reasoning", thought)
                    if piece and call_end >= 0:
                        # Past the call: normally just the end of the turn. The
                        # model is let finish so the server still reports usage,
                        # unless it starts writing the tool's answer itself.
                        chunks += 1
                        content += piece
                    elif piece:
                        chunks += 1
                        if ttft == 0.0:
                            ttft = time.time() - started
                        parts.append(piece)
                        if not self.server_stop:
                            content += piece
                            if CALL_END in content[-len(piece) - len(CALL_END):]:
                                ends = content.count(CALL_END)
                            if ends >= self.multi:
                                # the last call allowed is closed: stop right after it
                                call_end = len(cut_after_call(content, self.multi))
                                piece = piece[:len(piece) - (len(content) - call_end)]
                        if on_delta and piece:
                            on_delta("content", piece)
                if call_end >= 0 and len(content) - call_end > OVERRUN_CHARS:
                    finish = "stop"
                    break
                if cap and not parts and reasoning_chars > cap:
                    # Still inside <think> with the budget nearly gone: let it
                    # run and the whole reply comes back empty. Cut here so the
                    # loop can ask again with the thinking turned down, instead
                    # of waiting minutes for nothing.
                    finish = "think_cap"
                    break
        except Cancelled:
            finish = "cancelled"
        except (OSError, ValueError, http.client.HTTPException) as e:
            if cancelled():          # esc came while the error was on its way
                finish = "cancelled"
            else:
                # the tunnel or the server went away mid-reply (engine crash, tunnel drop)
                raise LLMError("connection lost mid-reply (%s: %s)" % (type(e).__name__, e),
                           transient=True)
        elapsed = time.time() - started
        if calls and CALL_END not in "".join(parts):
            call_text, complete = calls_to_text(calls, self._dialect, self.multi)
            if parts and "".join(parts).strip():
                call_text = "\n" + call_text
            parts.append(call_text)
            if on_delta:
                on_delta("content", call_text)
            if not complete:
                finish = "length"
        if not completion_tokens:      # server sent no usage -> estimate from chunks
            completion_tokens = chunks
        if not reasoning_tokens and reasoning_chars:
            reasoning_tokens = max(1, reasoning_chars // 4)
        return self._close("".join(parts), finish, self.multi), Usage(
            prompt_tokens, completion_tokens, reasoning_tokens, finish, ttft, elapsed,
            "".join(thoughts), cached_tokens)

    @staticmethod
    def _close(text, finish=None, limit=1):
        """The stop sequence eats the closing tag, so put it back - but only when
        generation actually stopped there. On finish_reason "length" the call was
        cut mid-argument, and closing it would turn a truncated call into one that
        parses cleanly with its last argument silently missing."""
        text = cut_after_call(text, limit)
        if finish == "length":
            return text
        if text.count("<tool_call>") > text.count("</tool_call>"):
            return text + "</tool_call>"
        return text
