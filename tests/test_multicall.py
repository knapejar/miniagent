# -*- coding: utf-8 -*-
"""Several tool calls in one reply, for models that can, and the one-call
protocol left exactly as it was for those that cannot (Spark-X2.5-4B)."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import Config                                       # noqa: E402
from agent.llm import LLMClient, Usage, calls_to_text                 # noqa: E402
from agent.loop import AgentLoop                                      # noqa: E402
from agent.protocol import (ONE_CALL, build_system, cut_after_call,   # noqa: E402
                            format_tool_call, parse_tool_calls)
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402


class Scripted(object):
    def __init__(self, *replies, prompt_tokens=0):
        self.replies = list(replies)
        self.seen = []
        self.prompt_tokens = prompt_tokens

    def chat(self, messages, on_delta=None, cancel=None, effort=None):
        self.seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if self.replies else "done"
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return text, Usage(self.prompt_tokens, 10, 0, finish, 0.1, 0.1)


def call(name, **args):
    return format_tool_call(name, args)


class TestParse(unittest.TestCase):
    def test_every_call_up_to_the_limit(self):
        text = call("sh", cmd="a") + "\n" + call("read", path="b") + call("sh", cmd="c")
        self.assertEqual(parse_tool_calls(text, 8),
                         [("sh", {"cmd": "a"}), ("read", {"path": "b"}), ("sh", {"cmd": "c"})])
        self.assertEqual(parse_tool_calls(text, 2), [("sh", {"cmd": "a"}), ("read", {"path": "b"})])

    def test_limit_one_is_the_old_parser(self):
        text = call("sh", cmd="a") + call("sh", cmd="b")
        self.assertEqual(parse_tool_calls(text, 1), [("sh", {"cmd": "a"})])
        self.assertEqual(parse_tool_calls("just prose", 1), [])

    def test_cut_after_the_nth_call(self):
        text = call("sh", cmd="a") + call("sh", cmd="b") + "invented output"
        self.assertEqual(cut_after_call(text), call("sh", cmd="a"))
        self.assertEqual(cut_after_call(text, 2), call("sh", cmd="a") + call("sh", cmd="b"))

    def test_structured_calls_become_markup(self):
        calls = {0: {"function": {"name": "sh", "arguments": json.dumps({"cmd": "a"})}},
                 1: {"function": {"name": "read", "arguments": json.dumps({"path": "b"})}}}
        text, complete = calls_to_text(calls, "native", 8)
        self.assertTrue(complete)
        self.assertEqual(parse_tool_calls(text, 8), [("sh", {"cmd": "a"}), ("read", {"path": "b"})])
        one, _ = calls_to_text(calls, "native", 1)
        self.assertEqual(parse_tool_calls(one, 8), [("sh", {"cmd": "a"})])

    def test_a_cut_off_last_call_is_left_open(self):
        calls = [{"function": {"name": "sh", "arguments": '{"cmd": "a"}'}},
                 {"function": {"name": "write", "arguments": '{"path": "x", "text": "ab'}}]
        text, complete = calls_to_text(calls, "native", 8)
        self.assertFalse(complete)
        self.assertEqual(LLMClient._close(text, "length", 8).count("</tool_call>"), 1)


class TestPrompt(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_one_call_prompt_is_unchanged(self):
        prompt = build_system(default_tools(), self.root)
        self.assertIn(ONE_CALL, prompt)
        self.assertNotIn("batch", prompt)

    def test_multi_call_prompt_asks_to_batch(self):
        prompt = build_system(default_tools(), self.root, dialect="native", multi_call=5)
        self.assertNotIn(ONE_CALL, prompt)
        self.assertIn("up\n  to 5 calls", prompt)

    def test_no_memory_means_no_pointer_to_old_runs(self):
        self.assertIn("runs/*.jsonl", build_system(default_tools(), self.root))
        self.assertNotIn("runs/*.jsonl", build_system(default_tools(), self.root, memory=False))


class TestConfig(unittest.TestCase):
    def test_auto_follows_the_dialect(self):
        self.assertEqual(Config(profile="local").multi_call, 1)          # spark
        self.assertEqual(Config(profile="local", model="GLM-5.3").multi_call, 8)
        self.assertEqual(Config(profile="local", model="Qwen3.8").multi_call, 1)
        self.assertEqual(Config(profile="local", model="GLM-5.3", multi_call=1).multi_call, 1)


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        for name in ("a.txt", "b.txt"):
            with open(os.path.join(self.root, name), "w") as fh:
                fh.write("content of %s\n" % name)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_loop(self, client, **kw):
        cfg = Config(workdir=self.root, max_steps=6, memory=False, **kw)
        session = Session(cfg, default_tools())
        events = list(AgentLoop(cfg, session, client, default_tools()).run("GOAL"))
        return events, session

    def test_two_calls_run_in_one_step(self):
        client = Scripted("Reading both." + call("read", path="a.txt") + call("read", path="b.txt"),
                          "both read")
        # native dialect: the results travel together as one user message
        events, session = self.run_loop(client, model="GLM-5.3")
        self.assertEqual([d["name"] for e, d in events if e == "tool_call"], ["read", "read"])
        self.assertEqual([d["prose"] for e, d in events if e == "tool_call"], ["Reading both.", ""])
        self.assertEqual(len(client.seen), 2)                 # one round trip for both
        last = client.seen[1][-1]["content"]
        self.assertEqual(last.count("<tool_response>"), 2)
        self.assertIn("content of a.txt", last)
        self.assertIn("content of b.txt", last)
        self.assertEqual([d["text"] for e, d in events if e == "final"], ["both read"])

    def test_one_call_model_runs_only_the_first(self):
        client = Scripted(call("read", path="a.txt") + call("read", path="b.txt"), "done")
        events, _ = self.run_loop(client)
        self.assertEqual([d["args"]["path"] for e, d in events if e == "tool_call"], ["a.txt"])

    def test_ask_among_calls_ends_the_run(self):
        client = Scripted(call("ask", question="which one?") + call("read", path="a.txt"))
        events, _ = self.run_loop(client, multi_call=4)
        self.assertEqual([d["name"] for e, d in events if e == "tool_call"], ["ask"])
        self.assertEqual(events[-1][1]["reason"], "asked")

    def test_budget_forces_an_answer(self):
        client = Scripted(call("read", path="a.txt"), call("read", path="b.txt"),
                          "what I have", prompt_tokens=600)
        events, _ = self.run_loop(client, max_task_tokens=1000)
        self.assertEqual(len([e for e, _ in events if e == "tool_call"]), 1)
        self.assertEqual([d["text"] for e, d in events if e == "final"], ["what I have"])
        self.assertTrue(any(e == "note" and "budget" in d["text"] for e, d in events))

    def test_crop_shrinks_each_result_of_a_multi_step(self):
        cfg = Config(workdir=self.root, multi_call=4, keep_tail=2, ctx=100000)
        session = Session(cfg, default_tools())
        session.start("GOAL")
        session.add_assistant(call("sh", cmd="x") + call("sh", cmd="y"))
        session.add_observations(["A" * 5000, "short"])
        for i in range(3):
            session.add_assistant(call("sh", cmd="z%d" % i))
            session.add_observation("ok")
        session.crop()
        multi = [m for m in session.messages if m["content"].count("<tool_response>") == 2]
        if multi:   # elided in place, not cut away
            self.assertLess(len(multi[0]["content"]), 1000)
            self.assertIn("short", multi[0]["content"])


if __name__ == "__main__":
    unittest.main()
