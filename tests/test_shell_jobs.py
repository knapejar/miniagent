# -*- coding: utf-8 -*-
"""A command that outlives a checkpoint: wait / background / kill / ctrl+b."""
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.tools.shell as shell                                     # noqa: E402
from agent.config import Config                                       # noqa: E402
from agent.llm import Usage                                           # noqa: E402
from agent.loop import AgentLoop                                      # noqa: E402
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402

SLOW = "ping -n %d 127.0.0.1 >nul & echo finished"


@unittest.skipUnless(os.name == "nt", "cmd.exe only")
class TestShellCheckpoints(unittest.TestCase):
    def setUp(self):
        self.saved = shell.CHECKPOINTS, shell.STALL
        shell.CHECKPOINTS, shell.STALL = (1, 2, 3, 4), 5
        self.session = Session(Config(workdir=tempfile.gettempdir()), default_tools())
        self.sh, self.wait = shell.ShellTool(), shell.WaitTool()

    def tearDown(self):
        shell.kill_all(self.session)
        shell.CHECKPOINTS, shell.STALL = self.saved

    def test_fast_command_returns_directly(self):
        self.assertTrue(self.sh.run(self.session, cmd="echo hi").startswith("exit=0"))

    def test_slow_command_asks_until_last_warning_then_stalls(self):
        out = self.sh.run(self.session, cmd=SLOW % 30)
        self.assertTrue(out.startswith("[RUNNING] sh1"), out)
        seen = [self.wait.run(self.session, job="sh1", action="wait") for _ in range(4)]
        self.assertTrue(seen[2].startswith("[LAST WARNING]"), seen)
        self.assertIn("stalled", seen[3])
        self.assertEqual(self.session.shell_jobs, {})

    def test_wait_until_done(self):
        self.sh.run(self.session, cmd=SLOW % 2)
        out = self.wait.run(self.session, job="sh1", action="wait")
        if out.startswith("[RUNNING]"):
            out = self.wait.run(self.session, job="sh1", action="wait")
        self.assertIn("finished", out)

    def test_kill(self):
        self.sh.run(self.session, cmd=SLOW % 30)
        self.assertTrue(self.wait.run(self.session, job="sh1", action="kill").startswith("killed"))
        self.assertEqual(self.session.shell_jobs, {})

    def test_ctrl_b_moves_it_to_background(self):
        self.sh.run(self.session, cmd=SLOW % 3)
        threading.Timer(0.2, self.session.background_key.set).start()
        out = self.wait.run(self.session, job="sh1", action="wait")
        self.assertIn("ctrl+b", out)
        done = self.session.jobs.drain(wait=5)
        self.assertIn("finished", done[0][2])

    def test_loop_backgrounds_an_undecided_command(self):
        class Client(object):
            replies = ["<tool_call>sh<arg_key>cmd</arg_key><arg_value>%s</arg_value>"
                       "</tool_call>" % (SLOW % 3),
                       "<tool_call>plan<arg_key>text</arg_key><arg_value>x</arg_value>"
                       "</tool_call>",
                       "all done", "all done"]

            def chat(self, messages, on_delta=None, cancel=None):
                return self.replies.pop(0), Usage(10, 10, 0, "stop", 0.1, 0.1)

        cfg = Config(workdir=tempfile.gettempdir(), max_steps=5)
        loop = AgentLoop(cfg, self.session, Client(), default_tools())
        events = list(loop.run("GOAL"))
        notes = [d["text"] for name, d in events if name == "note"]
        self.assertTrue(any("no decision was made" in n for n in notes), notes)
        self.assertTrue(any("background result" in n for n in notes), notes)


if __name__ == "__main__":
    unittest.main()
