# -*- coding: utf-8 -*-
"""Client for an OpenAI-compatible endpoint (LM Studio, llama-server, vLLM...).

Streams the reply chunk by chunk so tokens/s can be shown live, and so a cancel
request stops generation mid-token instead of after the whole reply.
No dependencies beyond the standard library.
"""
import http.client
import json
import threading
import time
import urllib.error
import urllib.request

from .protocol import cut_after_call, format_tool_call

# Statuses a tunnel or the kaggle-tpu-lab proxy answers with while the route to
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
                 finish_reason=None, ttft=0.0, elapsed=0.0, reasoning=""):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason
        self.ttft = ttft              # time to first token
        self.elapsed = elapsed
        self.reasoning = reasoning    # the reasoning text itself, kept for the history

    @property
    def tps(self):
        return self.completion_tokens / self.elapsed if self.elapsed > 0 else 0.0


def calls_to_text(calls, dialect="qwen"):
    """vLLM started with --enable-auto-tool-choice parses the model's tool call
    out of the answer even when the request lists no tools, and sends it as
    structured `tool_calls` with the text removed. Turn the first call back into
    the markup protocol.py reads. Returns (text, complete): arguments that are
    not valid JSON mean the call was cut off, so it is left unclosed."""
    if not calls:
        return "", True
    call = calls[min(calls)] if isinstance(calls, dict) else calls[0]
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
        if getattr(self.cfg, "profile", "local") == "kaggle":
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

    def _watch(self, response, cancel):
        """Close the socket when esc arrives, so a stream that has gone quiet
        stops blocking the read instead of waiting for the next chunk."""
        done = threading.Event()
        if cancel is None:
            return done
        def run():
            while not done.wait(0.1):
                if cancel.is_set():
                    response.close()
                    return
        threading.Thread(target=run, daemon=True).start()
        return done

    def _request(self, messages, stream, cancel=None, effort=None):
        data = json.dumps(self._body(messages, stream, effort)).encode("utf-8")
        retries = getattr(self.cfg, "retries", 0) or 0
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                self.cfg.url.rstrip("/") + "/chat/completions", data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + (self.cfg.api_key or "none"),
                         "ngrok-skip-browser-warning": "1",
                         # Ignored by every other backend; wafer.ai refuses
                         # without it unless the account allows retention.
                         "Wafer-ZDR": "required"})
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
            if cancel is not None and cancel.wait(RETRY_WAIT * (attempt + 1)):
                raise error
            if cancel is None:
                time.sleep(RETRY_WAIT * (attempt + 1))

    # ---------------------------------------------------------------- calls
    def chat(self, messages, on_delta=None, cancel=None, effort=None):
        """Return (text, Usage).

        `on_delta(kind, text)` receives chunks as they arrive, where kind is
        "content" or "reasoning". `cancel` is a threading.Event; when it is set
        the stream is abandoned and whatever arrived so far is returned.
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
            call_text, complete = calls_to_text(message["tool_calls"], self._dialect)
            text = (text.rstrip() + "\n" if text.strip() else "") + call_text
            if not complete:
                finish = "length"
        if not self.server_stop:
            text = cut_after_call(text)
        if on_delta and thought:
            on_delta("reasoning", thought)
        if on_delta and text:
            on_delta("content", text)
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        elapsed = time.time() - started
        return self._close(text, finish), Usage(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            details.get("reasoning_tokens", 0), finish, elapsed, elapsed, thought)

    def _chat_stream(self, messages, on_delta, cancel=None, effort=None):
        started = time.time()
        ttft = 0.0
        parts, thoughts, reasoning_chars, chunks = [], [], 0, 0
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        finish = None
        calls = {}             # index -> {"name", "arguments"} from structured tool_calls
        content = ""           # the answer so far, kept only for the client-side stop
        call_end = -1          # where </tool_call> ended in `content`
        cancelled = lambda: cancel is not None and cancel.is_set()
        try:
            cap = self.think_cap_chars if effort != "none" else 0
            with self._request(messages, True, cancel, effort) as response:
                watching = self._watch(response, cancel)
                try:
                    for raw in response:
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
                                        call_end = content.index(CALL_END) + len(CALL_END)
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
                finally:
                    watching.set()
        except Cancelled:
            finish = "cancelled"
        except (OSError, ValueError, http.client.HTTPException) as e:
            if cancelled():          # our own watcher closed the socket
                finish = "cancelled"
            else:
                # the tunnel or the server went away mid-reply (engine crash, tunnel drop)
                raise LLMError("connection lost mid-reply (%s: %s)" % (type(e).__name__, e),
                           transient=True)
        elapsed = time.time() - started
        if calls and CALL_END not in "".join(parts):
            call_text, complete = calls_to_text(calls, self._dialect)
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
        return self._close("".join(parts), finish), Usage(
            prompt_tokens, completion_tokens, reasoning_tokens, finish, ttft, elapsed,
            "".join(thoughts))

    @staticmethod
    def _close(text, finish=None):
        """The stop sequence eats the closing tag, so put it back - but only when
        generation actually stopped there. On finish_reason "length" the call was
        cut mid-argument, and closing it would turn a truncated call into one that
        parses cleanly with its last argument silently missing."""
        text = cut_after_call(text)
        if finish == "length":
            return text
        if "<tool_call>" in text and "</tool_call>" not in text:
            return text + "</tool_call>"
        return text
