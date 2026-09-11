# -*- coding: utf-8 -*-
"""Client for an OpenAI-compatible endpoint (LM Studio, llama-server, vLLM...).

Streams the reply chunk by chunk so tokens/s can be shown live, and so a cancel
request stops generation mid-token instead of after the whole reply.
No dependencies beyond the standard library.
"""
import json
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """Server-side failure. `overflow` means the prompt exceeded the context."""

    def __init__(self, message, status=None):
        RuntimeError.__init__(self, message)
        self.status = status
        low = message.lower()
        self.overflow = ("context" in low and "exceed" in low) or "too long" in low


class Usage(object):
    def __init__(self, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                 finish_reason=None, ttft=0.0, elapsed=0.0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.finish_reason = finish_reason
        self.ttft = ttft              # time to first token
        self.elapsed = elapsed

    @property
    def tps(self):
        return self.completion_tokens / self.elapsed if self.elapsed > 0 else 0.0


class LLMClient(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tool_role = "tool"       # probed at runtime: does the server accept it?

    # ---------------------------------------------------------- request body
    def _body(self, messages, stream):
        body = {"model": self.cfg.model, "messages": messages, "stream": stream,
                "temperature": self.cfg.temperature, "top_p": self.cfg.top_p,
                "top_k": self.cfg.top_k, "max_tokens": self.cfg.max_tokens,
                "stop": ["</tool_call>"]}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if self.cfg.effort != "default":
            # LM Studio ignores chat_template_kwargs (measured) and honours only
            # reasoning_effort; kwargs stay for other backends.
            body["reasoning_effort"] = self.cfg.effort
            if self.cfg.effort == "none":
                body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _request(self, messages, stream):
        request = urllib.request.Request(
            self.cfg.url.rstrip("/") + "/chat/completions",
            data=json.dumps(self._body(messages, stream)).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.cfg.api_key})
        try:
            return urllib.request.urlopen(request, timeout=self.cfg.timeout)
        except urllib.error.HTTPError as e:
            raise LLMError("HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:500]),
                           status=e.code)
        except urllib.error.URLError as e:
            raise LLMError("cannot reach %s (%s)" % (self.cfg.url, e.reason))

    # ---------------------------------------------------------------- calls
    def chat(self, messages, on_delta=None, cancel=None):
        """Return (text, Usage).

        `on_delta(kind, text)` receives chunks as they arrive, where kind is
        "content" or "reasoning". `cancel` is a threading.Event; when it is set
        the stream is abandoned and whatever arrived so far is returned.
        """
        if self.cfg.stream:
            return self._chat_stream(messages, on_delta, cancel)
        return self._chat_once(messages, on_delta)

    def _chat_once(self, messages, on_delta):
        started = time.time()
        with self._request(messages, False) as response:
            data = json.loads(response.read().decode("utf-8"))
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if on_delta and text:
            on_delta("content", text)
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        elapsed = time.time() - started
        finish = choice.get("finish_reason")
        return self._close(text, finish), Usage(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            details.get("reasoning_tokens", 0), finish, elapsed, elapsed)

    def _chat_stream(self, messages, on_delta, cancel=None):
        started = time.time()
        ttft = 0.0
        parts, reasoning_chars, chunks = [], 0, 0
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        finish = None
        with self._request(messages, True) as response:
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
                    if thought:
                        reasoning_chars += len(thought)
                        chunks += 1
                        if ttft == 0.0:
                            ttft = time.time() - started
                        if on_delta:
                            on_delta("reasoning", thought)
                    if piece:
                        chunks += 1
                        if ttft == 0.0:
                            ttft = time.time() - started
                        parts.append(piece)
                        if on_delta:
                            on_delta("content", piece)
        elapsed = time.time() - started
        if not completion_tokens:      # server sent no usage -> estimate from chunks
            completion_tokens = chunks
        if not reasoning_tokens and reasoning_chars:
            reasoning_tokens = max(1, reasoning_chars // 4)
        return self._close("".join(parts), finish), Usage(
            prompt_tokens, completion_tokens, reasoning_tokens, finish, ttft, elapsed)

    @staticmethod
    def _close(text, finish=None):
        """The stop sequence eats the closing tag, so put it back - but only when
        generation actually stopped there. On finish_reason "length" the call was
        cut mid-argument, and closing it would turn a truncated call into one that
        parses cleanly with its last argument silently missing."""
        if finish == "length":
            return text
        if "<tool_call>" in text and "</tool_call>" not in text:
            return text + "</tool_call>"
        return text
