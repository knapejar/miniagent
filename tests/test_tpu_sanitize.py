# -*- coding: utf-8 -*-
"""Requests that crash the TPU engine (tpu_inference 0.28) never reach vLLM: the kernel's front
server and launch.py's proxy share sanitize_request() from the Qwen kernel."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "kaggle-tpu-lab"))

import launch  # noqa: E402


def adapt(path, data, model="qwen3.8-27b"):
    return json.loads(launch.adapt_body(path, json.dumps(data).encode(), model))


class TestSanitize(unittest.TestCase):
    def test_claude_code_title_request(self):
        r = adapt("/v1/messages?beta=true", {
            "model": "q", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
            "output_config": {"format": {"type": "json_schema", "schema": {}}, "effort": "max"},
            "tool_choice": {"type": "tool", "name": "t"},
            "tools": [{"name": "t", "input_schema": {}, "strict": True}], "temperature": 0})
        self.assertEqual(r["output_config"], {"effort": "xhigh"})
        self.assertEqual(r["tool_choice"], {"type": "auto"})
        self.assertNotIn("strict", r["tools"][0])
        self.assertEqual((r["temperature"], r["top_k"]), (1.0, 1))

    def test_chat_completions_crash_fields(self):
        r = adapt("/v1/chat/completions", {
            "model": "q", "stream": True, "logprobs": True, "n": 3, "echo": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                      {"type": "video_url", "video_url": {"url": "x"}}]}],
            "response_format": {"type": "json_object"}, "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "f", "strict": True, "parameters": {}}}],
            "prompt_logprobs": 1, "allowed_token_ids": [1], "seed": 3, "min_p": 0.1, "logit_bias": {"1": 2}})
        for key in ("response_format", "prompt_logprobs", "allowed_token_ids", "seed", "min_p", "logit_bias"):
            self.assertNotIn(key, r)
        self.assertEqual((r["echo"], r["tool_choice"], r["top_logprobs"], r["n"], r["stream"]),
                         (False, "auto", 1, 1, True))
        self.assertNotIn("strict", r["tools"][0]["function"])
        self.assertTrue(r["messages"][0]["content"][-1]["text"].startswith("[video removed"))

    def test_completions(self):
        r = adapt("/v1/completions", {"model": "q", "prompt": "x", "logprobs": 0, "echo": True, "best_of": 2})
        self.assertEqual(r, {"model": "q", "prompt": "x", "echo": False})

    def test_ordinary_requests_pass_untouched(self):
        plain = json.dumps({"model": "q", "messages": [{"role": "user", "content": "hi"}], "tool_choice": "auto",
                            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
                            "temperature": 0.7, "max_tokens": 100}).encode()
        self.assertEqual(launch.adapt_body("/v1/chat/completions", plain, "qwen3.8-27b"), plain)
        self.assertEqual(launch.adapt_body("/v1/models", b"", "qwen3.8-27b"), b"")
        glm = json.dumps({"response_format": {"type": "json_object"}}).encode()
        self.assertEqual(launch.adapt_body("/v1/chat/completions", glm, "glm-5.3-flash"), glm)

    def test_kernel_front_server_uses_it(self):
        with open(launch.KERNEL_SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("body, dropped = sanitize_request(self.path, self._body())", src)
        self.assertIn('os.environ["VLLM_ENFORCE_STRICT_TOOL_CALLING"] = "0"', src)


if __name__ == "__main__":
    unittest.main()
