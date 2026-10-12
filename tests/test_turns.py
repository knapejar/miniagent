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
        self.efforts = getattr(self, "efforts", []) + [effort]
        reply = self.replies.pop(0)
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return text, Usage(0, 10, 0, finish, 0.1, 0.1)    # 0: no calibration


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


class TestContext(unittest.TestCase):
    """Room is made cheapest first; a finished plan item starts the next clean."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def session(self, **kw):
        cfg = Config(workdir=self.root, ctx=kw.get("ctx", 20000), keep_tail=4, crop_at=0.7,
                     remind_every=0, memory=False, checkpoint_at=kw.get("checkpoint_at", 0.3))
        s = Session(cfg, default_tools())
        s.start("GOAL")
        return s

    def step(self, s, reply, observation, reasoning=""):
        s.add_assistant(reply, reasoning)
        s.add_observation(observation)

    def test_stale_reads_and_old_reasoning_go_before_anything_is_cut(self):
        s = self.session()
        self.step(s, call("read", path="a.py"), "A" * 6000, "r" * 10000)
        self.step(s, call("edit", path="a.py", old="x", new="y"), "edited a.py", "r" * 10000)
        self.step(s, call("read", path="b.py"), "B" * 6000, "r" * 10000)
        self.step(s, call("sh", cmd="python t.py"), "exit=0\n" + "t" * 3000)
        for i in range(2):
            self.step(s, call("sh", cmd="echo %d" % i), "exit=0 %d" % i)
        s.crop()
        self.assertEqual(s.last_crop, "elided")
        contents = [m["content"] for m in s.messages]
        self.assertTrue(any("[stale: this read of a.py" in c for c in contents))
        self.assertTrue(any(c.endswith("B" * 300 + "\n...[5700 chars dropped to save room - run "
                                       "the call again if you need them]</tool_response>")
                            for c in contents))
        self.assertFalse(any(m.get("reasoning_content") for m in s.messages))
        self.assertTrue(any(call("read", path="a.py") == c for c in contents))   # the call stays
        self.assertFalse(any("context cropped" in c for c in contents))

    def test_when_shrinking_is_not_enough_the_middle_is_cut(self):
        s = self.session(ctx=4000)
        for i in range(10):
            self.step(s, call("sh", cmd="echo %d" % i), "out %d " % i + "x" * 400)
        s.crop()
        self.assertEqual(s.last_crop, "cut")
        self.assertIn("context cropped", s.messages[3]["content"])

    def test_a_checkpoint_keeps_only_the_plan_call(self):
        s = self.session()
        self.step(s, call("read", path="a.py"), "A" * 100)
        self.step(s, call("plan", text="1. read - DONE"), "plan updated")
        s.set_plan("1. read - DONE")
        s.checkpoint()
        self.assertEqual([m["role"] for m in s.messages[3:]], ["user", "assistant", s.tool_role])
        self.assertIn("a.py: read 1x", s.messages[3]["content"])
        self.assertIn("1. read - DONE", s.messages[2]["content"])

    def test_the_loop_checkpoints_when_a_plan_item_turns_done(self):
        put(self.root, "big.py", "x = 1\n" * 1500)
        client = Scripted(call("plan", text="1. read big.py\n2. say done"),
                          call("read", path="big.py"),
                          call("plan", text="1. read big.py - DONE\n2. say done"), "done")
        cfg = Config(workdir=self.root, ctx=6000, max_steps=6, memory=False, checkpoint_at=0.3)
        session = Session(cfg, default_tools())
        events = list(AgentLoop(cfg, session, client, default_tools()).run("GOAL"))
        self.assertTrue(any(e == "crop" for e, _ in events))
        self.assertIn("big.py: read 1x", session.messages[3]["content"])
        self.assertNotIn("x = 1\n" * 50, "".join(m["content"] for m in client.seen[-1]))

    def test_no_checkpoint_while_the_run_is_small_or_when_off(self):
        for share in (0.3, 0):
            client = Scripted(call("plan", text="1. a - DONE"), "done")
            cfg = Config(workdir=self.root, ctx=100000, max_steps=4, memory=False,
                         checkpoint_at=share)
            events = list(AgentLoop(cfg, Session(cfg, default_tools()), client,
                                    default_tools()).run("GOAL"))
            self.assertFalse(any(e == "crop" for e, _ in events))

    def test_an_empty_reply_leaves_no_spiral_behind(self):
        client = Scripted(("<think>" + "hmm " * 2000 + "</think>", "think_cap"), "done")
        cfg = Config(workdir=self.root, max_steps=4, memory=False)
        session = Session(cfg, default_tools())
        list(AgentLoop(cfg, session, client, default_tools()).run("GOAL"))
        self.assertFalse(any("hmm" in (m.get("reasoning_content") or "") for m in session.messages))


class TestReading(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.session = Session(Config(workdir=self.root), default_tools())
        self.tools = dict((t.name, t) for t in default_tools())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def read(self, **args):
        return self.tools["read"].run(self.session, **args)

    def test_a_cut_read_outlines_what_it_did_not_show(self):
        body = "".join("def f%d():\n%s" % (i, "    x = 1\n" * 20) for i in range(30))
        put(self.root, "big.py", body)
        out = self.read(path="big.py")
        self.assertIn("continue with start=201", out)
        self.assertIn("Outline of the lines not shown:", out)
        self.assertIn("\tdef f29():", out)
        self.assertNotIn("\tdef f0():\n", out.split("Outline")[1])   # already shown above
        self.assertLessEqual(len(out), self.tools["read"].max_chars)

    def test_a_whole_read_has_no_outline(self):
        put(self.root, "small.py", "def f():\n    return 1\n")
        self.assertNotIn("Outline", self.read(path="small.py"))

    def test_markdown_is_outlined_by_headings(self):
        put(self.root, "doc.md", "".join("# Part %d\n%s" % (i, "text\n" * 50) for i in range(8)))
        out = self.read(path="doc.md", count=20)
        self.assertIn("\t# Part 7", out)

    def test_several_paths_in_one_read(self):
        put(self.root, "a.py", "A = 1\n")
        put(self.root, "b.py", "B = 2\n")
        out = self.read(path="a.py\nb.py\nmissing.py")
        self.assertIn("=== a.py\n    1\tA = 1", out)
        self.assertIn("=== b.py\n    1\tB = 2", out)
        self.assertIn("=== missing.py\nerror:", out)
        self.assertLessEqual(len(out), self.tools["read"].max_chars)


class TestUndo(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self._home = os.environ.get("MINIAGENT_HOME")
        os.environ["MINIAGENT_HOME"] = self.home
        self.root = tempfile.mkdtemp()
        self.session = Session(Config(workdir=self.root), default_tools())
        self.session.start("GOAL")
        self.tools = dict((t.name, t) for t in default_tools())

    def tearDown(self):
        if self._home is None:
            os.environ.pop("MINIAGENT_HOME", None)
        else:
            os.environ["MINIAGENT_HOME"] = self._home
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.home, ignore_errors=True)

    def tool(self, name, **args):
        return self.tools[name].run(self.session, **args)

    def text(self, name):
        with open(os.path.join(self.root, name), encoding="utf-8") as fh:
            return fh.read()

    def test_edit_undo_takes_back_one_change_at_a_time(self):
        put(self.root, "a.py", "x = 1\n")
        self.tool("edit", path="a.py", old="x = 1", new="x = 2")
        self.tool("edit", path="a.py", old="x = 2", new="x = 3")
        self.assertIn("undone", self.tool("edit", path="a.py", undo="true"))
        self.assertEqual(self.text("a.py"), "x = 2\n")
        self.tool("edit", path="a.py", undo="true")
        self.assertEqual(self.text("a.py"), "x = 1\n")
        self.assertTrue(self.tool("edit", path="a.py", undo="true").startswith("error"))

    def test_undo_of_a_new_file_removes_it(self):
        self.tool("write", path="new.py", text="y = 1\n")
        self.assertIn("removed", self.tool("edit", path="new.py", undo="true"))
        self.assertFalse(os.path.exists(os.path.join(self.root, "new.py")))

    def test_the_original_is_copied_to_disk_once(self):
        put(self.root, "a.py", "x = 1\n")
        self.tool("edit", path="a.py", old="x = 1", new="x = 2")
        self.tool("write", path="a.py", text="x = 3\n")
        copies = os.listdir(self.session.backup_dir)
        self.assertEqual(len(copies), 1)
        with open(os.path.join(self.session.backup_dir, copies[0]), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "x = 1\n")

    def test_the_slash_command_puts_the_whole_task_back(self):
        import contextlib
        import io
        import miniagent
        put(self.root, "a.py", "x = 1\n")
        self.tool("edit", path="a.py", old="x = 1", new="x = 2")
        self.tool("write", path="b.py", text="new\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            miniagent.handle_command("/undo", self.session.cfg, self.session,
                                     type("R", (), {"s": __import__("ui.ansi", fromlist=["x"]).Style(False)})(),
                                     None, None)
        self.assertEqual(self.text("a.py"), "x = 1\n")
        self.assertFalse(os.path.exists(os.path.join(self.root, "b.py")))
        self.assertEqual(self.session.undo, {})


if __name__ == "__main__":
    unittest.main()
