# -*- coding: utf-8 -*-
"""A server outage (vLLM engine crash + in-place restart, tunnel drop) must not end the
session: the client reports it as transient and the loop waits it out."""
import io
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.loop as loop_mod                                          # noqa: E402
from agent.config import Config                                       # noqa: E402
from agent.llm import LLMClient, LLMError, Usage                      # noqa: E402
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402


class BrokenStream(object):
    """Yields some lines, then the connection is reset (what the Kaggle proxy did)."""

    def __init__(self, lines, error):
        self.lines, self.error = lines, error

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for line in self.lines:
            yield line
        raise self.error


class TestClientTransient(unittest.TestCase):
    def client(self):
        return LLMClient(Config(profile="kaggle", retries=0))

    def test_connection_reset_on_connect_is_transient(self):
        with mock.patch("urllib.request.urlopen", side_effect=ConnectionResetError(10054, "reset")):
            with self.assertRaises(LLMError) as ctx:
                self.client().chat([{"role": "user", "content": "hi"}])
        self.assertTrue(ctx.exception.transient)

    def test_engine_dead_500_is_transient_other_500_is_not(self):
        def http_error(text):
            return urllib.error.HTTPError("u", 500, "x", {}, io.BytesIO(text.encode()))
        for text, expected in (('{"error":{"message":"EngineCore encountered an issue."}}', True),
                               ('{"error":{"message":"bad schema"}}', False)):
            with mock.patch("urllib.request.urlopen", side_effect=http_error(text)):
                with self.assertRaises(LLMError) as ctx:
                    self.client().chat([{"role": "user", "content": "hi"}])
            self.assertEqual(ctx.exception.transient, expected, text)

    def test_reset_mid_stream_is_transient(self):
        client = self.client()
        line = ("data: " + json.dumps({"choices": [{"delta": {"content": "par"}}]}) + "\n").encode()
        client._request = lambda *a, **kw: BrokenStream([line], ConnectionResetError(10054, "reset"))
        with self.assertRaises(LLMError) as ctx:
            client.chat([])
        self.assertTrue(ctx.exception.transient)

    def test_error_event_in_stream_is_raised(self):
        client = self.client()
        line = ("data: " + json.dumps({"error": {"message": "EngineCore encountered an issue.",
                                                  "code": 500}}) + "\n").encode()
        client._request = lambda *a, **kw: BrokenStream([line], StopIteration())
        with self.assertRaises(LLMError) as ctx:
            client.chat([])
        self.assertTrue(ctx.exception.transient)


class FlakyClient(object):
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def chat(self, messages, on_delta=None, cancel=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMError("connection lost mid-reply (ConnectionResetError)", transient=True)
        return "All done.", Usage(10, 5, 0, "stop", 0.1, 0.1)


class TestLoopWaitsOutOutage(unittest.TestCase):
    def run_loop(self, client, outage_wait):
        workdir = os.path.join(os.environ.get("TEMP", "."), "miniagent_outage")
        os.makedirs(workdir, exist_ok=True)
        cfg = Config(workdir=workdir, max_steps=3, outage_wait=outage_wait,
                     memory=False)   # this is about the outage, not the memory pass
        session = Session(cfg, default_tools())
        with mock.patch.object(loop_mod, "OUTAGE_POLL", 0):
            return list(loop_mod.AgentLoop(cfg, session, client, default_tools()).run("GOAL"))

    def test_transient_errors_are_retried_and_the_task_finishes(self):
        client = FlakyClient(failures=2)
        events = self.run_loop(client, outage_wait=1800)
        self.assertEqual(client.calls, 3)
        self.assertFalse(any(n == "end" and d.get("reason") == "error" for n, d in events))
        self.assertEqual(sum(1 for n, d in events if n == "note" and "unavailable" in d["text"]), 2)

    def test_local_profile_does_not_wait(self):
        client = FlakyClient(failures=1)
        events = self.run_loop(client, outage_wait=0)
        self.assertEqual(client.calls, 1)
        self.assertTrue(any(n == "end" and d.get("reason") == "error" for n, d in events))


if __name__ == "__main__":
    unittest.main()
