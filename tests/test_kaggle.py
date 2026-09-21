# -*- coding: utf-8 -*-
"""Qwen3.8 on a Kaggle TPU (kaggle-tpu-lab): the qwen tool-call dialect, the
kaggle profile's request body, and the startup checks in agent/kaggle.py.

Run with:  python -m unittest discover -s tests -v"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import kaggle, llm, web                                    # noqa: E402
from agent.config import Config, parse_args                           # noqa: E402
from agent.llm import LLMClient, LLMError                             # noqa: E402
from agent.protocol import (build_system, dialect_for, format_tool_call,   # noqa: E402
                            malformed, parse_tool_call, strip_think)
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402

QWEN_CALL = ("Let me write it.\n<tool_call>\n<function=write>\n<parameter=path>\n"
             "app.py\n</parameter>\n<parameter=text>\ndef f():\n    return 1\n\n"
             "</parameter>\n</function>\n</tool_call>")


class TestQwenDialect(unittest.TestCase):
    def test_parses_the_native_qwen_format(self):
        name, args = parse_tool_call(QWEN_CALL)
        self.assertEqual(name, "write")
        self.assertEqual(args["path"], "app.py")

    def test_keeps_indentation_and_inner_blank_lines(self):
        _, args = parse_tool_call(QWEN_CALL)
        self.assertEqual(args["text"], "def f():\n    return 1\n")

    def test_missing_closing_parameter_tag(self):
        name, args = parse_tool_call(
            "<tool_call>\n<function=sh>\n<parameter=cmd>\ndir /b\n</function>\n</tool_call>")
        self.assertEqual((name, args), ("sh", {"cmd": "dir /b"}))

    def test_unterminated_call(self):
        name, args = parse_tool_call("<tool_call>\n<function=read>\n<parameter=path>\nx.txt")
        self.assertEqual((name, args["path"]), ("read", "x.txt"))

    def test_roundtrip(self):
        args = {"path": "a.py", "old": "  x = 1", "new": "  x = 2\n"}
        self.assertEqual(parse_tool_call(format_tool_call("edit", args, "qwen")),
                         ("edit", args))

    def test_repeated_parameter_is_several_values(self):
        _, args = parse_tool_call("<tool_call><function=websearch><parameter=query>a"
                                  "</parameter><parameter=query>b</parameter></function>"
                                  "</tool_call>")
        self.assertEqual(args["query"], "a\nb")

    def test_nested_qwen_markup_is_malformed(self):
        _, args = parse_tool_call("<tool_call><function=sh><parameter=cmd>"
                                  "<function=read><parameter=path>x</parameter>"
                                  "</function></tool_call>")
        self.assertTrue(malformed(args))

    def test_spark_format_still_parses(self):
        self.assertEqual(parse_tool_call(
            "<tool_call>sh<arg_key>cmd</arg_key><arg_value>ver</arg_value></tool_call>"),
            ("sh", {"cmd": "ver"}))

    def test_reasoning_is_dropped_before_parsing(self):
        self.assertEqual(parse_tool_call(strip_think("hmm</think>" + QWEN_CALL))[0], "write")

    def test_system_prompt_teaches_the_format(self):
        prompt = build_system(default_tools(), r"C:\ws", dialect="qwen")
        self.assertIn("<function=example_function_name>", prompt)
        self.assertIn('{"type":"function","function":{"name":"sh"', prompt)
        self.assertIn("AT MOST 5 lines", prompt)
        self.assertLess(len(prompt) / 3.4, 1550)

    def test_spark_prompt_is_unchanged(self):
        prompt = build_system(default_tools(), r"C:\ws")
        self.assertIn("## Tools\nYou have access to the following functions:\n<tools>", prompt)
        self.assertNotIn("<function=", prompt)

    def test_dialect_from_model_name(self):
        self.assertEqual(dialect_for("Qwen3.8-27B"), "qwen")
        self.assertEqual(dialect_for("spark-x2.5-4b"), "spark")


class TestKaggleProfile(unittest.TestCase):
    def body(self, **kw):
        return LLMClient(Config(profile="kaggle", **kw))._body([], True)

    def test_defaults(self):
        cfg = Config(profile="kaggle")
        self.assertEqual(cfg.dialect, "qwen")
        self.assertEqual(cfg.model, "qwen3.8-27b")
        self.assertEqual(cfg.effort, "medium")
        self.assertTrue(cfg.url.endswith(":8080/v1"))

    def test_local_profile_is_unchanged(self):
        cfg = Config()
        self.assertEqual((cfg.dialect, cfg.ctx, cfg.temperature, cfg.effort),
                         ("spark", 65536, 1.0, "low"))
        body = LLMClient(cfg)._body([], True)
        self.assertEqual(body["stop"], ["</tool_call>"])
        self.assertEqual(body["reasoning_effort"], "low")

    def test_sampling_is_left_to_the_model_generation_config(self):
        body = self.body()
        for name in ("temperature", "top_p", "top_k"):
            self.assertNotIn(name, body)
        self.assertEqual(self.body(temperature=0.6)["temperature"], 0.6)

    def test_no_server_stop_and_no_top_level_effort(self):
        body = self.body()
        self.assertNotIn("stop", body)
        self.assertNotIn("reasoning_effort", body)

    def test_effort_goes_through_the_chat_template(self):
        self.assertEqual(self.body(effort="low")["chat_template_kwargs"],
                         {"reasoning_effort": "low"})
        self.assertEqual(self.body(effort="high")["chat_template_kwargs"],
                         {"reasoning_effort": "xhigh"})
        self.assertEqual(self.body(effort="none")["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertNotIn("chat_template_kwargs", self.body(effort="default"))

    def test_command_line(self):
        cfg, task = parse_args(["--kaggle", "--effort", "low", "--ctx", "32768", "do it"])
        self.assertEqual((cfg.profile, cfg.effort, cfg.ctx, task),
                         ("kaggle", "low", 32768, "do it"))
        cfg, _ = parse_args([])
        self.assertEqual((cfg.profile, cfg.model), ("local", "spark-x2.5-4b"))

    def test_vllm_overflow_message(self):
        self.assertTrue(LLMError("HTTP 400: This model's maximum context length is 131072 "
                                 "tokens. However, you requested 140000 tokens").overflow)

    def test_tool_message_is_not_wrapped_twice(self):
        session = Session(Config(profile="kaggle", workdir=tempfile.gettempdir()),
                          default_tools())
        session.add_observation("ok")
        self.assertEqual(session.messages[-1]["content"], "ok")
        session.downgrade_tool_role()
        self.assertEqual(session.messages[-1], {"role": "user",
                                                "content": "<tool_response>ok</tool_response>"})


class FakeStream(object):
    def __init__(self, events):
        self.lines = [("data: " + json.dumps(e) + "\n").encode() for e in events]
        self.lines.append(b"data: [DONE]\n")
        self.read_lines = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for line in self.lines:
            self.read_lines += 1
            yield line


class TestClientSideStop(unittest.TestCase):
    def run_stream(self, reasoning, content, usage=True):
        events = [{"choices": [{"delta": {"reasoning_content": reasoning}}]}]
        events += [{"choices": [{"delta": {"content": content[i:i + 4]}}]}
                   for i in range(0, len(content), 4)]
        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        if usage:
            events.append({"choices": [], "usage": {"prompt_tokens": 900,
                                                    "completion_tokens": 50}})
        stream = FakeStream(events)
        client = LLMClient(Config(profile="kaggle"))
        client._request = lambda *a, **kw: stream
        shown = []
        text, used = client.chat([], on_delta=lambda kind, t: shown.append((kind, t)))
        return text, used, shown, stream

    def test_stops_after_the_call_not_inside_reasoning(self):
        text, used, shown, _ = self.run_stream("the format ends with </tool_call> ok",
                                               QWEN_CALL)
        self.assertEqual(text, QWEN_CALL)
        self.assertEqual(used.prompt_tokens, 900)

    def test_invented_tool_response_is_cut_and_the_stream_abandoned(self):
        tail = "\n<tool_response>\nwrote it\n</tool_response>\n" * 5
        text, used, shown, stream = self.run_stream("", QWEN_CALL + tail)
        self.assertEqual(text, QWEN_CALL)
        self.assertNotIn("tool_response", "".join(t for k, t in shown if k == "content"))
        self.assertLess(stream.read_lines, len(stream.lines))

    def test_plain_answer_passes_through(self):
        text, _, _, _ = self.run_stream("", "All done, 3 files changed.")
        self.assertEqual(text, "All done, 3 files changed.")


class TestStructuredToolCalls(unittest.TestCase):
    """vLLM with --enable-auto-tool-choice returns the call as tool_calls, not text
    (seen on the real Kaggle server: finish_reason tool_calls, no content)."""

    def client(self, events):
        stream = FakeStream(events)
        client = LLMClient(Config(profile="kaggle"))
        client._request = lambda *a, **kw: stream
        return client

    def test_streamed_tool_call_becomes_markup(self):
        args = json.dumps({"path": "app.py", "text": "def f():\n    return 1\n"})
        events = [{"choices": [{"delta": {"reasoning": "write the file"}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                   "type": "function", "function": {"name": "write"}}]}}]}]
        events += [{"choices": [{"delta": {"tool_calls": [{"index": 0,
                   "function": {"arguments": args[i:i + 7]}}]}}]}
                   for i in range(0, len(args), 7)]
        events.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        shown = []
        text, used = self.client(events).chat([], on_delta=lambda k, t: shown.append((k, t)))
        self.assertEqual(parse_tool_call(text),
                         ("write", {"path": "app.py", "text": "def f():\n    return 1\n"}))
        self.assertIn("<function=write>", "".join(t for k, t in shown if k == "content"))
        self.assertGreater(used.completion_tokens, 0)

    def test_non_string_arguments_are_rendered_as_json(self):
        events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "name": "read", "arguments": json.dumps({"path": "a.txt", "limit": 40})}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}]
        text, _ = self.client(events).chat([])
        self.assertEqual(parse_tool_call(text), ("read", {"path": "a.txt", "limit": "40"}))

    def test_cut_off_arguments_stay_a_truncated_call(self):
        events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "name": "write", "arguments": '{"path": "a.py", "text": "def'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]}]
        text, used = self.client(events).chat([])
        self.assertNotIn("</tool_call>", text)
        self.assertEqual(used.finish_reason, "length")

    def test_non_stream_message_tool_calls(self):
        client = LLMClient(Config(profile="kaggle", stream=False))
        body = {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {
                "name": "sh", "arguments": json.dumps({"cmd": "dir"})}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        class Response(object):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(body).encode()
        client._request = lambda *a, **kw: Response()
        text, _ = client.chat([])
        self.assertEqual(parse_tool_call(text), ("sh", {"cmd": "dir"}))


class TestRetries(unittest.TestCase):
    def setUp(self):
        self.urlopen, self.wait = llm.urllib.request.urlopen, llm.RETRY_WAIT
        llm.RETRY_WAIT = 0

    def tearDown(self):
        llm.urllib.request.urlopen, llm.RETRY_WAIT = self.urlopen, self.wait

    def fail_then(self, errors):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            if len(calls) <= len(errors):
                raise errors[len(calls) - 1]
            return "response"
        llm.urllib.request.urlopen = urlopen
        return calls

    @staticmethod
    def http(code):
        return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b"tunnel down"))

    def test_tunnel_errors_are_retried(self):
        calls = self.fail_then([self.http(502), urllib.error.URLError("refused")])
        client = LLMClient(Config(profile="kaggle"))
        self.assertEqual(client._request([], True, threading.Event()), "response")
        self.assertEqual(len(calls), 3)

    def test_a_bad_request_is_not(self):
        calls = self.fail_then([self.http(400)])
        with self.assertRaises(LLMError):
            LLMClient(Config(profile="kaggle"))._request([], True)
        self.assertEqual(len(calls), 1)

    def test_local_profile_does_not_retry(self):
        calls = self.fail_then([self.http(502)])
        with self.assertRaises(LLMError):
            LLMClient(Config())._request([], True)
        self.assertEqual(len(calls), 1)


class TestKaggleSetup(unittest.TestCase):
    def test_kernel_phase(self):
        self.assertEqual(kaggle.kernel_phase(None)[0], "unknown")
        self.assertEqual(kaggle.kernel_phase([])[0], "starting")
        self.assertEqual(kaggle.kernel_phase([{"phase": "install"},
                                              {"phase": "compiling"}])[0], "starting")
        live = kaggle.kernel_phase([{"phase": "ready", "max_model_len": 262144},
                                    {"phase": "heartbeat"}, {"phase": "cmd-result"}])
        self.assertEqual(live[0], "live")
        self.assertEqual(kaggle.kernel_phase([{"phase": "ready"},
                                              {"phase": "auto-shutdown"}])[0], "dead")

    def test_searxng_is_turned_off_only_on_a_clash(self):
        env = {}
        self.assertTrue(kaggle._avoid_searxng_clash("http://127.0.0.1:8080/v1", env))
        self.assertEqual(env["MINIAGENT_SEARX"], "off")
        self.assertFalse(kaggle._avoid_searxng_clash("http://127.0.0.1:8081/v1", {}))
        custom = {"MINIAGENT_SEARX": "http://127.0.0.1:8888"}
        self.assertFalse(kaggle._avoid_searxng_clash("http://127.0.0.1:8080/v1", custom))
        self.assertEqual(custom["MINIAGENT_SEARX"], "http://127.0.0.1:8888")

    def test_search_skips_a_disabled_searxng(self):
        os.environ["MINIAGENT_SEARX"] = "off"
        try:
            self.assertIsNone(web.searxng_base())
            self.assertEqual(web.searxng("q", 3), [])
        finally:
            del os.environ["MINIAGENT_SEARX"]
        self.assertTrue(web.searxng_base())

    def test_launcher_from_environment(self):
        folder = tempfile.mkdtemp()
        open(os.path.join(folder, "launch.py"), "w").close()
        self.assertEqual(kaggle.find_launcher({"KAGGLE_TPU_LAB": folder}),
                         os.path.abspath(os.path.join(folder, "launch.py")))

    def test_state_key_and_server_limits(self):
        path = os.path.join(tempfile.mkdtemp(), "state.json")
        with open(path, "w") as fh:
            json.dump({"kernel": "me/qwen38-tpu-serve", "api_key": "sk-abcdefghijklmnop"}, fh)
        original = kaggle.probe, kaggle.port_open
        kaggle.probe = lambda url, key, timeout=0: (
            {"id": "qwen3.8-27b", "max_model_len": 32768, "ids": ["qwen3.8-27b"]}, None)
        kaggle.port_open = lambda *a, **kw: True
        try:
            cfg = Config(profile="kaggle", url="http://127.0.0.1:18080/v1")
            self.assertTrue(kaggle.setup(cfg, say=lambda t: None, state_path=path))
        finally:
            kaggle.probe, kaggle.port_open = original
        self.assertEqual(cfg.api_key, "sk-abcdefghijklmnop")
        self.assertEqual(cfg.ctx, 32768)
        self.assertIn("me/qwen38-tpu-serve", cfg.backend_note)

    def test_a_queued_kernel_is_reported_not_probed(self):
        path = os.path.join(tempfile.mkdtemp(), "state.json")
        with open(path, "w") as fh:
            json.dump({"kernel": "me/k", "topic": "t", "api_key": "sk-x"}, fh)
        original = kaggle.kernel_events, kaggle.probe
        kaggle.kernel_events = lambda topic: []
        kaggle.probe = lambda *a, **kw: self.fail("must not probe a queued kernel")
        said = []
        searx = os.environ.get("MINIAGENT_SEARX")
        try:
            ok = kaggle.setup(Config(profile="kaggle"), say=said.append, state_path=path)
        finally:
            kaggle.kernel_events, kaggle.probe = original
            if searx is None:
                os.environ.pop("MINIAGENT_SEARX", None)
            else:
                os.environ["MINIAGENT_SEARX"] = searx
        self.assertFalse(ok)
        self.assertTrue(any("not live yet" in s for s in said))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestThinkCapStream(unittest.TestCase):
    """The spiral that started this: a model that never leaves <think> and returns
    an empty reply after minutes of generation. Cut it as soon as the reasoning
    passes its share of the budget, so the loop can ask again with less thinking."""

    def _run(self, reasoning_chunks, **cfg_kw):
        events = [{"choices": [{"delta": {"reasoning_content": c}}]}
                  for c in reasoning_chunks]
        events.append({"choices": [{"delta": {"content": "late answer"},
                                    "finish_reason": "stop"}]})
        stream = FakeStream(events)
        client = LLMClient(Config(max_tokens=100, **cfg_kw))   # cap = 100*4*0.6 = 240 chars
        client._request = lambda *a, **kw: stream
        return client.chat([]) + (stream,)

    def test_runaway_reasoning_is_cut(self):
        text, used, stream = self._run(["x" * 100] * 6)
        self.assertEqual(used.finish_reason, "think_cap")
        self.assertEqual(text, "")
        self.assertLess(stream.read_lines, len(stream.lines))   # abandoned early

    def test_reasoning_within_the_cap_is_left_alone(self):
        text, used, _ = self._run(["x" * 50, "x" * 50])
        self.assertEqual(used.finish_reason, "stop")
        self.assertEqual(text, "late answer")

    def test_cap_off_lets_it_think_as_long_as_it_likes(self):
        text, used, _ = self._run(["x" * 100] * 6, think_cap=0)
        self.assertEqual(text, "late answer")
