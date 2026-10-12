# -*- coding: utf-8 -*-
"""How a turn is shaped for a larger change: orientation before the first
step, and a check before an answer about changed code counts as final."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import workspace                                           # noqa: E402
from agent.config import Config                                       # noqa: E402
from agent.llm import Usage                                           # noqa: E402
from agent.loop import AgentLoop                                      # noqa: E402
from agent.protocol import build_system, format_tool_call             # noqa: E402
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402


def put(root, name, text=""):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class Scripted(object):
    """A model that says the given replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, on_delta=None, cancel=None, effort=None):
        self.seen.append(list(messages))
        return self.replies.pop(0), Usage(10, 10, 0, "stop", 0.1, 0.1)


def call(name, **args):
    return format_tool_call(name, args)


class TestOrientation(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_the_system_prompt_lists_the_working_directory(self):
        put(self.root, "app.py")
        put(self.root, os.path.join("pkg", "core.py"))
        put(self.root, os.path.join("node_modules", "x.js"))
        prompt = build_system(default_tools(), self.root)
        self.assertIn("app.py", prompt)
        self.assertIn("pkg/  (1: core.py)", prompt)
        self.assertNotIn("node_modules", prompt)

    def test_an_empty_directory_says_so(self):
        self.assertIn("The working directory is empty.", workspace.snapshot(self.root))

    def test_the_test_command_is_named(self):
        put(self.root, os.path.join("tests", "test_a.py"))
        self.assertEqual(workspace.test_command(self.root), "python -m unittest discover -s tests")
        put(self.root, "conftest.py")
        self.assertEqual(workspace.test_command(self.root), "python -m pytest -q")
        put(self.root, "package.json", '{"scripts": {"test": "jest"}}')
        self.assertEqual(workspace.test_command(self.root), "npm test")
        self.assertIn("Tests run with: npm test", workspace.snapshot(self.root))

    def test_no_test_command_is_guessed(self):
        put(self.root, "notes.txt")
        self.assertEqual(workspace.test_command(self.root), "")


class TestCheckBeforeFinal(unittest.TestCase):
    """The README case: a model wrote 108 asserts, never ran them, said ALL OK."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_loop(self, client):
        cfg = Config(workdir=self.root, max_steps=8, memory=False)
        session = Session(cfg, default_tools())
        events = list(AgentLoop(cfg, session, client, default_tools()).run("GOAL"))
        return events, session

    def test_an_answer_after_an_unrun_change_is_held_back_once(self):
        client = Scripted(call("write", path="a.py", text="print(6*7)\n"), "done",
                          call("sh", cmd="python a.py"), "done, it prints 42")
        events, session = self.run_loop(client)
        finals = [d["text"] for e, d in events if e == "final"]
        self.assertEqual(finals, ["done, it prints 42"])
        checks = [m["content"] for m in session.messages if m["content"].startswith("[CHECK]")]
        self.assertEqual(len(checks), 1)
        self.assertIn("a.py", checks[0])

    def test_a_change_that_was_run_passes(self):
        client = Scripted(call("write", path="a.py", text="print(1)\n"),
                          call("sh", cmd="python a.py"), "done")
        events, _ = self.run_loop(client)
        self.assertEqual([d["text"] for e, d in events if e == "final"], ["done"])
        self.assertFalse(any(e == "note" and "held back" in d["text"] for e, d in events))

    def test_a_failing_last_run_is_asked_about_once(self):
        client = Scripted(call("write", path="a.py", text="raise SystemExit(3)\n"),
                          call("sh", cmd="python a.py"), "done", "it still exits with 3")
        events, _ = self.run_loop(client)
        self.assertEqual([d["text"] for e, d in events if e == "final"], ["it still exits with 3"])

    def test_changing_notes_needs_no_run(self):
        client = Scripted(call("write", path="notes.md", text="# x\n"), "done")
        events, _ = self.run_loop(client)
        self.assertEqual([d["text"] for e, d in events if e == "final"], ["done"])

    def test_looking_at_a_file_is_not_running_it(self):
        client = Scripted(call("write", path="a.py", text="x = 1\n"), call("sh", cmd="type a.py"),
                          "done", "done")
        events, _ = self.run_loop(client)
        self.assertTrue(any(e == "note" and "held back" in d["text"] for e, d in events))


if __name__ == "__main__":
    unittest.main()
