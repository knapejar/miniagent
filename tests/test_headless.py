# -*- coding: utf-8 -*-
"""Headless mode end to end: miniagent as a child process, driven over
stream-json the way Claude Code's print mode is, against a scripted
OpenAI-compatible server and a small MCP server."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.headless import text_of, visible       # noqa: E402

MCP_SERVER = r'''
import json, sys
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m:
        continue
    if m["method"] == "initialize":
        r = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
             "serverInfo": {"name": "calc", "version": "1"}}
    elif m["method"] == "tools/list":
        r = {"tools": [{"name": "add", "description": "Add two numbers.",
                        "inputSchema": {"type": "object", "required": ["a", "b"],
                                        "properties": {"a": {"type": "number"},
                                                       "b": {"type": "number"}}}}]}
    elif m["method"] == "tools/call":
        a = m["params"]["arguments"]
        r = {"content": [{"type": "text", "text": "sum=%s" % (a["a"] + a["b"])}]}
    else:
        r = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}) + "\n")
    sys.stdout.flush()
'''


def tool_reply(name, args):
    return {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}]}


class Model(object):
    """Replies in order; each is a text answer or a tool call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model.requests.append({"body": request, "headers": dict(self.headers)})
            reply = model.replies.pop(0) if model.replies else "NOTHING"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def chunk(delta, finish=None):
                data = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: %s\n\n" % json.dumps(data)).encode())

            if isinstance(reply, dict):
                chunk({"content": "Let me check. "})
                chunk(reply, "tool_calls")
            else:
                for i in range(0, len(reply), 4):
                    chunk({"content": reply[i:i + 4]})
                chunk({}, "stop")
            usage = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 7}}
            self.wfile.write(("data: %s\n\n" % json.dumps(usage)).encode())
            self.wfile.write(b"data: [DONE]\n\n")
    return Handler


class TestHelpers(unittest.TestCase):
    def test_visible_hides_reasoning_calls_and_half_tags(self):
        self.assertEqual(visible("<think>hm</think>Hello"), "Hello")
        self.assertEqual(visible("Hello <tool_call>sh"), "Hello ")
        self.assertEqual(visible("Hello <tool_c"), "Hello ")
        self.assertEqual(visible("reasoning only</think>Answer"), "Answer")

    def test_text_of_blocks(self):
        self.assertEqual(text_of("hi"), "hi")
        self.assertIn("image", text_of([{"type": "text", "text": "a"}, {"type": "image"}]))


class TestHeadless(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.work = tempfile.mkdtemp()
        self.model = Model([])
        self.procs = []
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(self.model))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        os.makedirs(os.path.join(self.home, "profiles"))
        with open(os.path.join(self.home, "profiles", "test.json"), "w") as fh:
            json.dump({"url": "http://127.0.0.1:%d/v1" % self.server.server_port,
                       "headers": {"X-Test": "yes"}, "api_key_env": "TEST_KEY_UNSET",
                       "profiles": {"t": {"model": "test-model"}}}, fh)
        with open(os.path.join(self.home, "test.key"), "w") as fh:
            fh.write("secret-key")
        self.mcp = os.path.join(self.home, "calc.py")
        with open(self.mcp, "w") as fh:
            fh.write(MCP_SERVER)

    def tearDown(self):
        for proc in self.procs:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe and not pipe.closed:
                    pipe.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.home, ignore_errors=True)
        shutil.rmtree(self.work, ignore_errors=True)

    def start(self, *extra):
        env = dict(os.environ, MINIAGENT_HOME=self.home, MINIAGENT_SEARX="off")
        argv = [sys.executable, os.path.join(ROOT, "miniagent.py"), "-p",
                "--input-format", "stream-json", "--output-format", "stream-json",
                "--include-partial-messages", "--verbose", "--profile", "t",
                "--no-memory", "-C", self.work] + list(extra)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, cwd=self.work)
        self.procs.append(proc)
        return proc

    @staticmethod
    def ask(proc, text):
        proc.stdin.write((json.dumps({"type": "user", "message": {
            "role": "user", "content": [{"type": "text", "text": text}]}}) + "\n").encode())
        proc.stdin.flush()
        events = []
        for raw in iter(proc.stdout.readline, b""):
            events.append(json.loads(raw))
            if events[-1]["type"] == "result":
                break
        return events

    def test_a_turn_with_a_tool_call(self):
        cmd = "echo hello-from-sh"
        self.model.replies = [tool_reply("sh", {"cmd": cmd}), "All done, it said hello."]
        proc = self.start("--session-id", "11111111-2222-3333-4444-555555555555",
                          "--append-system-prompt", "EXTRA RULE 42")
        events = self.ask(proc, "say hello")
        proc.stdin.close()
        proc.wait(timeout=20)

        init = events[0]
        self.assertEqual((init["type"], init["subtype"]), ("system", "init"))
        self.assertEqual(init["session_id"], "11111111-2222-3333-4444-555555555555")
        uses = [b for e in events if e["type"] == "assistant"
                for b in e["message"]["content"] if b["type"] == "tool_use"]
        self.assertEqual(uses[0]["name"], "sh")
        self.assertEqual(uses[0]["input"]["cmd"], cmd)
        results = [b for e in events if e["type"] == "user" for b in e["message"]["content"]]
        self.assertEqual(results[0]["tool_use_id"], uses[0]["id"])
        self.assertIn("hello-from-sh", results[0]["content"])
        deltas = "".join(e["event"]["delta"].get("text", "") for e in events
                         if e["type"] == "stream_event"
                         and e["event"]["type"] == "content_block_delta")
        self.assertIn("All done", deltas)
        self.assertNotIn("<tool_call>", deltas)
        final = events[-1]
        self.assertEqual(final["subtype"], "success")
        self.assertEqual(final["result"], "All done, it said hello.")
        self.assertEqual(final["usage"]["input_tokens"], 200)
        # the profile's headers and key reach the server, the appended prompt the model
        first = self.model.requests[0]
        self.assertEqual(first["headers"].get("X-Test"), "yes")
        self.assertEqual(first["headers"].get("Authorization"), "Bearer secret-key")
        self.assertIn("EXTRA RULE 42", first["body"]["messages"][0]["content"])
        saved = os.listdir(os.path.join(self.home, "projects", os.listdir(
            os.path.join(self.home, "projects"))[0], "sessions"))
        self.assertIn("11111111-2222-3333-4444-555555555555.json", saved)

    def test_resume_continues_the_conversation(self):
        sid = "abcdefab-0000-0000-0000-000000000001"
        self.model.replies = ["The word is PINEAPPLE."]
        proc = self.start("--session-id", sid)
        self.ask(proc, "remember the word pineapple")
        proc.stdin.close()
        proc.wait(timeout=20)

        self.model.replies = ["PINEAPPLE"]
        proc = self.start("--resume", sid)
        events = self.ask(proc, "what was the word?")
        proc.stdin.close()
        proc.wait(timeout=20)
        self.assertEqual(events[-1]["session_id"], sid)
        sent = json.dumps(self.model.requests[-1]["body"]["messages"])
        self.assertIn("remember the word pineapple", sent)
        self.assertIn("what was the word?", sent)

    def test_resume_of_an_unknown_session_fails_like_claude(self):
        proc = self.start("--resume", "does-not-exist")
        out, err = proc.communicate(timeout=20)
        self.assertEqual(proc.returncode, 1)
        self.assertIn(b"No conversation found with session ID", err)

    def test_mcp_tools_are_offered_and_called(self):
        config = json.dumps({"mcpServers": {"calc": {"command": sys.executable,
                                                     "args": [self.mcp]}}})
        self.model.replies = [tool_reply("mcp__calc__add", {"a": 2, "b": 40}), "It is 42."]
        proc = self.start("--mcp-config", config)
        events = self.ask(proc, "add 2 and 40")
        proc.stdin.close()
        proc.wait(timeout=20)
        self.assertIn({"name": "calc", "status": "connected"}, events[0]["mcp_servers"])
        self.assertIn("mcp__calc__add", events[0]["tools"])
        offered = [t["function"]["name"] for t in self.model.requests[0]["body"]["tools"]]
        self.assertIn("mcp__calc__add", offered)
        results = [b for e in events if e["type"] == "user" for b in e["message"]["content"]]
        self.assertEqual(results[0]["content"], "sum=42")

    def test_disallowed_tools_are_not_offered(self):
        self.model.replies = ["ok"]
        proc = self.start("--disallowedTools", "websearch", "browse")
        self.ask(proc, "hi")
        proc.stdin.close()
        proc.wait(timeout=20)
        offered = [t["function"]["name"] for t in self.model.requests[0]["body"]["tools"]]
        self.assertNotIn("websearch", offered)
        self.assertNotIn("browse", offered)
        self.assertIn("read", offered)

    def test_one_shot_text_output(self):
        self.model.replies = ["Just the answer."]
        env = dict(os.environ, MINIAGENT_HOME=self.home, MINIAGENT_SEARX="off")
        out = subprocess.run([sys.executable, os.path.join(ROOT, "miniagent.py"), "-p",
                              "question", "--profile", "t", "--no-memory", "-C", self.work],
                             capture_output=True, env=env, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.decode().strip(), "Just the answer.")


if __name__ == "__main__":
    unittest.main()
