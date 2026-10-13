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
from test_kaggle import setUpModule, tearDownModule  # noqa: E402,F401 - the kaggle profile


class BrokenStream(object):
    """Yields some lines, then the connection is reset (what a tunnel proxy did)."""

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

    def chat(self, messages, on_delta=None, cancel=None, effort=None):
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


class EscDuringConnect(unittest.TestCase):
    """A server that accepts but never answers must not hold esc hostage."""

    def test_cancel_returns_while_urlopen_is_still_blocked(self):
        import threading, time
        from agent.llm import Cancelled
        cfg = Config(url="http://127.0.0.1:1/v1", model="m", timeout=30, retries=0)
        client = LLMClient(cfg, [])
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.time()
        with mock.patch("urllib.request.urlopen", lambda *a, **k: time.sleep(30)):
            with self.assertRaises(Cancelled):
                client._open(mock.Mock(), cancel)
        self.assertLess(time.time() - started, 5)

    def test_cancel_returns_while_a_reply_has_gone_quiet(self):
        """What a dropped wifi did: the reply started, then no byte and no reset.
        The read blocked for the whole timeout and nothing could stop it."""
        import socket
        import threading
        import time
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        held = []

        def serve():
            conn, _ = server.accept()
            held.append(conn)
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n8\r\ndata:{}\n\r\n")
        threading.Thread(target=serve, daemon=True).start()
        cfg = Config(url="http://127.0.0.1:%d/v1" % server.getsockname()[1], model="m",
                     timeout=30, retries=0)
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        started = time.time()
        try:
            _text, usage = LLMClient(cfg, []).chat([{"role": "user", "content": "hi"}],
                                                   cancel=cancel)
        finally:
            for conn in held + [server]:
                conn.close()
        self.assertEqual(usage.finish_reason, "cancelled")
        self.assertLess(time.time() - started, 5)

    def test_a_retry_is_announced_and_esc_ends_its_wait(self):
        import threading
        import time
        cfg = Config(url="http://127.0.0.1:1/v1", model="m", timeout=30, retries=2)
        cancel, heard = threading.Event(), []

        def on_delta(kind, piece):
            heard.append((kind, piece))
            cancel.set()                         # esc while it counts down
        started = time.time()
        with mock.patch("urllib.request.urlopen",
                        mock.Mock(side_effect=urllib.error.URLError("no route"))):
            _text, usage = LLMClient(cfg, []).chat([], on_delta, cancel)
        self.assertEqual(usage.finish_reason, "cancelled")
        self.assertLess(time.time() - started, 5)
        (kind, (error, wait)), = heard
        self.assertEqual(kind, "retry")
        self.assertIn("no route", error)
        self.assertIn("retry 1/2", error)
        self.assertGreater(wait, 0)

if __name__ == "__main__":
    unittest.main()
