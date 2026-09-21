# -*- coding: utf-8 -*-
"""Core tests.  Run with:  python -m unittest discover -s tests -v"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import Config                                       # noqa: E402
from agent.loop import clip                                           # noqa: E402
from agent.protocol import (PLAN_HEADER, build_system, format_tool_call,   # noqa: E402
                            parse_tool_call, strip_think)
from agent.secrets import has_secret, redact                          # noqa: E402
from agent.session import Session                                     # noqa: E402
from agent.tools import default_tools                                 # noqa: E402
from agent.tools.shell import looks_like_powershell                   # noqa: E402
from agent import paths                                               # noqa: E402

# Paid for in every turn of the model - a budget, not a style rule. Raised from
# 1100 when websearch and browse gained multi-target and background: 26 tokens
# that turn four sequential searches (25.7s) into one call (4.7s). Raised again
# to 1350 for the project-state convention rule (agent/protocol.py).
SYSTEM_PROMPT_TOKEN_BUDGET = 1350


class TestPaths(unittest.TestCase):
    """State belongs in ~/.miniagent, never in the directory being worked on."""

    def setUp(self):
        self._home = os.environ.get(paths.ENV_HOME)
        os.environ[paths.ENV_HOME] = os.path.join(os.path.abspath("."), "_home_for_tests")

    def tearDown(self):
        if self._home is None:
            os.environ.pop(paths.ENV_HOME, None)
        else:
            os.environ[paths.ENV_HOME] = self._home

    def test_slug_flattens_the_absolute_path(self):
        self.assertEqual(paths.slug(os.path.abspath(os.sep)).count(os.sep), 0)
        self.assertNotIn(":", paths.slug("."))
        self.assertNotIn(" ", paths.slug("."))

    def test_two_projects_get_two_slots(self):
        here, up = paths.project_dir("."), paths.project_dir("..")
        self.assertNotEqual(here, up)
        self.assertTrue(here.startswith(paths.home()))

    def test_memory_and_runs_sit_side_by_side(self):
        self.assertEqual(os.path.dirname(paths.runs_dir(".")), paths.project_dir("."))
        self.assertEqual(os.path.dirname(paths.memory_dir(".")), paths.project_dir("."))
        self.assertEqual(paths.memory_index("."),
                         os.path.join(paths.memory_dir("."), "INDEX.md"))

    def test_the_trace_never_lands_in_the_working_directory(self):
        cfg = Config(workdir=".")
        self.assertTrue(cfg.trace_dir.startswith(paths.home()))
        self.assertEqual(cfg.trace_dir, paths.runs_dir("."))

    def test_an_explicit_trace_dir_still_wins(self):
        self.assertEqual(Config(trace_dir="elsewhere").trace_dir, "elsewhere")

class TestProtocol(unittest.TestCase):
    def test_native_spark_format(self):
        name, args = parse_tool_call(
            "<tool_call>sh<arg_key>cmd</arg_key><arg_value>dir /b</arg_value></tool_call>")
        self.assertEqual(name, "sh")
        self.assertEqual(args, {"cmd": "dir /b"})

    def test_multiple_args_and_newlines(self):
        name, args = parse_tool_call(
            "<tool_call>write<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
            "<arg_key>text</arg_key><arg_value>l1\nl2</arg_value></tool_call>")
        self.assertEqual(name, "write")
        self.assertEqual(args["text"], "l1\nl2")

    def test_hermes_json_fallback(self):
        parsed = parse_tool_call('<tool_call>{"name":"read","arguments":{"path":"x"}}</tool_call>')
        self.assertEqual(parsed, ("read", {"path": "x"}))

    def test_unterminated_call(self):
        parsed = parse_tool_call("<tool_call>sh<arg_key>cmd</arg_key><arg_value>ver</arg_value>")
        self.assertEqual(parsed, ("sh", {"cmd": "ver"}))

    def test_no_call(self):
        self.assertIsNone(parse_tool_call("done, the file has 12 lines"))

    def test_roundtrip(self):
        text = format_tool_call("edit", {"path": "a.py", "old": "x", "new": "y"})
        self.assertEqual(parse_tool_call(text),
                         ("edit", {"path": "a.py", "old": "x", "new": "y"}))

    def test_strip_think(self):
        self.assertEqual(strip_think("reasoning</think>Done."), "Done.")
        self.assertEqual(strip_think("<think>x</think>Y"), "Y")
        self.assertEqual(parse_tool_call(strip_think(
            "thinking</think><tool_call>sh<arg_key>cmd</arg_key>"
            "<arg_value>ver</arg_value></tool_call>"))[0], "sh")

    def test_system_prompt_stays_within_budget(self):
        prompt = build_system(default_tools(), r"C:\ws")
        self.assertIn("<tools>", prompt)
        self.assertLess(len(prompt) / 3.4, SYSTEM_PROMPT_TOKEN_BUDGET)


class TestSecrets(unittest.TestCase):
    def test_github_token_is_masked(self):
        line = "token: gho_EXAMPLEexampleEXAMPLEexample000000"
        out = redact(line)
        self.assertNotIn("gho_EXAMPLE", out)
        self.assertIn("[REDACTED GitHub token]", out)

    def test_url_credentials_are_masked(self):
        out = redact("https://user:pass@github.com/x/y.git")
        self.assertNotIn("user:pass", out)
        self.assertIn("github.com/x/y.git", out)

    def test_plain_text_untouched(self):
        self.assertEqual(redact("nothing to see here"), "nothing to see here")

    def test_can_be_disabled(self):
        line = "gho_EXAMPLEexampleEXAMPLEexample000000"
        self.assertEqual(redact(line, enabled=False), line)

    def test_detector(self):
        self.assertTrue(has_secret("sk-abcdefghijklmnopqrstuvwxyz123456"))
        self.assertFalse(has_secret("just a sentence"))


class TestShellDialect(unittest.TestCase):
    def test_powershell_syntax_detected(self):
        for cmd in ['$env:TOKEN = "x"; gh repo create',
                    "Get-Process | Where-Object { $_.Name -like 'x' }",
                    "curl x | Select-String foo",
                    "Test-Path C:\\tmp"]:
            self.assertTrue(looks_like_powershell(cmd), cmd)

    def test_cmd_syntax_not_misrouted(self):
        for cmd in ["dir /b", "python fib.py", "git status", "echo %PATH%",
                    'powershell -Command "Get-Process"']:
            self.assertFalse(looks_like_powershell(cmd), cmd)


class TestClip(unittest.TestCase):
    def test_short_untouched(self):
        self.assertEqual(clip("abc", 100), "abc")

    def test_long_clipped(self):
        out = clip("A" * 10000, 3000)
        self.assertLess(len(out), 3120)
        self.assertIn("chars cut", out)
        self.assertTrue(out.startswith("AAA") and out.endswith("AAA"))


class TestSession(unittest.TestCase):
    def make(self, **kw):
        cfg = Config(workdir=os.environ.get("TEMP", "."), ctx=1000,
                     keep_tail=kw.get("keep_tail", 4), crop_at=0.7,
                     remind_every=kw.get("remind_every", 8))
        return Session(cfg, default_tools())

    def fill(self, session, count=14):
        for i in range(count):
            session.add_assistant("step %d " % i + "x" * 300)
            session.add_observation("out %d" % i)

    def test_layout(self):
        s = self.make()
        s.start("GOAL")
        self.assertEqual(s.messages[0]["role"], "system")
        self.assertEqual(s.messages[1]["content"], "GOAL")
        self.assertTrue(s.messages[2]["content"].startswith(PLAN_HEADER))

    def test_crop_keeps_head_and_tail(self):
        s = self.make()
        s.start("GOAL")
        s.set_plan("[x] headings DONE\n[ ] lists")
        self.fill(s)
        before = len(s.messages)
        dropped = s.crop()
        self.assertGreater(dropped, 0)
        self.assertLess(len(s.messages), before)
        self.assertEqual(s.messages[1]["content"], "GOAL")
        self.assertIn("headings DONE", s.messages[2]["content"])
        self.assertIn("context cropped", s.messages[3]["content"])
        self.assertIn("out 13", s.messages[-1]["content"])

    def test_plan_survives_two_crops(self):
        s = self.make()
        s.start("GOAL")
        s.set_plan("MEMORY")
        self.fill(s)
        s.crop()
        self.fill(s, 6)
        s.crop()
        self.assertIn("MEMORY", s.messages[2]["content"])

    def test_second_task_is_appended_not_overwritten(self):
        """The prefix in front of a new task has to stay byte-identical, or the
        server re-prefills the whole history instead of reusing its KV cache."""
        s = self.make()
        s.start("FIRST")
        self.fill(s, 3)
        before = list(s.messages)
        s.start("SECOND")
        self.assertEqual(s.messages[:len(before)], before)      # prefix untouched
        self.assertEqual(s.messages[-1]["content"], "SECOND")
        self.assertEqual(s.anchor, len(s.messages) - 1)
        self.assertEqual(s.goal, "SECOND")

    def test_crop_keeps_the_appended_task(self):
        s = self.make()
        s.start("FIRST")
        self.fill(s, 3)
        s.start("SECOND")
        self.fill(s)
        s.crop()
        self.assertEqual(s.messages[s.anchor]["content"], "SECOND")
        self.assertIn("context cropped", s.messages[3]["content"])
        self.assertIn("[GOAL] SECOND", s.reminder(1, 50))

    def test_anchor_index_survives_crop_from_the_tail(self):
        """An anchor still inside keep_tail is not cut, but its index shifts."""
        s = self.make()
        s.start("FIRST")
        self.fill(s)
        s.start("SECOND")
        self.fill(s, 2)
        s.crop()
        self.assertEqual(s.messages[s.anchor]["content"], "SECOND")

    def test_set_goal_replaces_the_current_task(self):
        s = self.make()
        s.start("FIRST")
        self.fill(s, 2)
        s.start("SECOND")
        s.set_goal("THIRD")
        self.assertEqual(s.messages[s.anchor]["content"], "THIRD")
        self.assertEqual(s.goal, "THIRD")
        self.assertEqual(s.messages[1]["content"], "FIRST")

    def test_reset_puts_the_anchor_back(self):
        s = self.make()
        s.start("FIRST")
        self.fill(s, 2)
        s.start("SECOND")
        s.reset()
        s.start("AGAIN")
        self.assertEqual(s.anchor, 1)
        self.assertEqual(s.messages[1]["content"], "AGAIN")
        self.assertTrue(s.messages[2]["content"].startswith(PLAN_HEADER))

    def test_calibration(self):
        s = self.make()
        s.start("GOAL")
        naive = s.estimated_tokens()
        s.calibrate(int(s.total_chars() / 3.0))     # server reports more tokens
        self.assertGreater(s.estimated_tokens(), naive)

    def test_reminder_schedule(self):
        s = self.make(remind_every=8)
        s.start("GOAL")
        self.assertEqual(s.reminder(3, 50), "")
        self.assertIn("[GOAL] GOAL", s.reminder(8, 50))
        self.assertIn("[step 16/50", s.reminder(16, 50))

    def test_reminder_after_crop(self):
        s = self.make()
        s.start("GOAL")
        s.cropped = True
        self.assertIn("[GOAL]", s.reminder(5, 50))
        self.assertEqual(s.reminder(5, 50), "")     # the flag resets

    def test_reminder_can_be_disabled(self):
        s = self.make(remind_every=0)
        s.start("GOAL")
        self.assertEqual(s.reminder(24, 50), "")


class TestTools(unittest.TestCase):
    def setUp(self):
        workdir = os.path.join(os.environ.get("TEMP", "."), "miniagent_tests")
        os.makedirs(workdir, exist_ok=True)
        cfg = Config(workdir=workdir)
        self.session = Session(cfg, default_tools())
        self.tools = {t.name: t for t in default_tools()}

    def test_shell_exit_code_and_output(self):
        out = self.tools["sh"].run(self.session, cmd="echo hello")
        self.assertIn("exit=0", out)
        self.assertIn("hello", out)

    def test_shell_propagates_exit_code(self):
        self.assertIn("exit=3", self.tools["sh"].run(self.session, cmd="exit /b 3"))

    def test_shell_multiline_and_cwd_persist(self):
        self.tools["sh"].run(self.session, cmd="mkdir sub 2>nul\ncd sub\necho inside")
        self.assertTrue(self.session.cwd.lower().endswith("sub"))

    def test_shell_literal_percent(self):
        out = self.tools["sh"].run(self.session, cmd='python -c "print(\'%%d\' %% 7)"')
        self.assertIn("7", out)

    def test_powershell_is_routed_automatically(self):
        out = self.tools["sh"].run(self.session, cmd='$env:MINI_T = "ok"; Write-Host $env:MINI_T')
        self.assertIn("shell=powershell", out)
        self.assertIn("ok", out)

    def test_file_roundtrip(self):
        self.tools["write"].run(self.session, path="t.py", text="print('a')\nprint(1+1)\n")
        self.assertIn("    1| print('a')", self.tools["read"].run(self.session, path="t.py"))
        self.assertIn("edited", self.tools["edit"].run(self.session, path="t.py",
                                                       old="1+1", new="2+2"))
        self.assertIn("2+2", self.tools["read"].run(self.session, path="t.py"))

    def test_edit_reports_missing_anchor(self):
        self.tools["write"].run(self.session, path="t2.py", text="abc")
        out = self.tools["edit"].run(self.session, path="t2.py", old="zzz", new="y")
        self.assertTrue(out.startswith("error"))

    def test_edit_rejects_identical_old_and_new(self):
        self.tools["write"].run(self.session, path="t3.py", text="abc")
        out = self.tools["edit"].run(self.session, path="t3.py", old="abc", new="abc")
        self.assertIn("identical", out)

    def test_read_missing_file(self):
        self.assertTrue(self.tools["read"].run(self.session, path="nope.txt").startswith("error"))

    def test_grep_finds_matches(self):
        self.tools["write"].run(self.session, path="g1.py", text="alpha\nbeta\ngamma\n")
        out = self.tools["grep"].run(self.session, pattern="bet", glob="g1.py")
        self.assertIn("g1.py:2", out)
        self.assertIn("beta", out)

    def test_grep_reports_no_matches(self):
        out = self.tools["grep"].run(self.session, pattern="zzz_not_here", glob="*.py")
        self.assertIn("no matches", out)

    def test_grep_rejects_bad_regex(self):
        self.assertTrue(self.tools["grep"].run(self.session, pattern="([").startswith("error"))

    def test_plan_tool_writes_pinned_slot(self):
        self.session.start("GOAL")
        self.tools["plan"].run(self.session, text="[x] done")
        self.assertIn("[x] done", self.session.messages[2]["content"])

    def test_risky_command_needs_approval(self):
        sh = self.tools["sh"]
        self.assertTrue(sh.needs_approval({"cmd": "del /s C:\\data"}))
        self.assertTrue(sh.needs_approval({"cmd": "gh auth token"}))
        self.assertFalse(sh.needs_approval({"cmd": "dir /b"}))


class TestCancellation(unittest.TestCase):
    def test_loop_stops_when_cancel_is_set(self):
        from agent.loop import AgentLoop

        class DeadClient(object):
            def chat(self, *a, **kw):
                raise AssertionError("the model must not be called after cancelling")

        cfg = Config(workdir=os.environ.get("TEMP", "."), max_steps=5)
        session = Session(cfg, default_tools())
        cancel = threading.Event()
        cancel.set()
        loop = AgentLoop(cfg, session, DeadClient(), default_tools(), cancel=cancel)
        events = [name for name, _ in loop.run("GOAL")]
        self.assertIn("end", events)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestGuards(unittest.TestCase):
    def test_failed_detection(self):
        from agent.guards import failed
        self.assertTrue(failed("exit=1  cwd=x\nboom"))
        self.assertTrue(failed("error: nope"))
        self.assertFalse(failed("exit=0  cwd=x\nfine"))

    def test_failure_memory_blocks_after_limit(self):
        from agent.guards import FailureMemory
        memory = FailureMemory(limit=3)
        for _ in range(3):
            memory.record("sh|git push", True)
        self.assertTrue(memory.blocked("sh|git push"))
        memory.record("sh|git push", False)
        self.assertFalse(memory.blocked("sh|git push"))

    def test_result_repeat(self):
        from agent.guards import ResultRepeat
        repeat = ResultRepeat(window=3)
        self.assertEqual(repeat.push("same"), "")
        self.assertEqual(repeat.push("same"), "")
        self.assertIn("identical result", repeat.push("same"))

    def test_result_repeat_ignores_varied_output(self):
        from agent.guards import ResultRepeat
        repeat = ResultRepeat(window=3)
        for text in ("a", "b", "c", "d"):
            self.assertEqual(repeat.push(text), "")


class TestValidation(unittest.TestCase):
    def setUp(self):
        workdir = os.path.join(os.environ.get("TEMP", "."), "miniagent_validate")
        os.makedirs(workdir, exist_ok=True)
        self.session = Session(Config(workdir=workdir), default_tools())
        self.tools = {t.name: t for t in default_tools()}

    def test_broken_python_is_reported_immediately(self):
        out = self.tools["write"].run(self.session, path="broken.py", text="def f(:\n")
        self.assertIn("SYNTAX ERROR", out)

    def test_valid_python_is_quiet(self):
        out = self.tools["write"].run(self.session, path="ok.py", text="x = 1\n")
        self.assertNotIn("SYNTAX ERROR", out)

    def test_broken_json_is_reported(self):
        self.assertIn("JSON ERROR",
                      self.tools["write"].run(self.session, path="b.json", text="{oops}"))

    def test_edit_that_breaks_a_file_is_reported(self):
        self.tools["write"].run(self.session, path="e.py", text="def f():\n    return 1\n")
        out = self.tools["edit"].run(self.session, path="e.py",
                                     old="def f():", new="def f(:")
        self.assertIn("SYNTAX ERROR", out)


class TestSecretEnv(unittest.TestCase):
    def test_collect_env_picks_credential_names(self):
        from agent.secrets import collect_env
        found = collect_env({"GH_TOKEN": "abcdefghij", "HOME": "/home/x",
                             "MY_API_KEY": "0123456789", "SHORT_TOKEN": "ab"})
        self.assertEqual(sorted(found), ["GH_TOKEN", "MY_API_KEY"])

    def test_registered_value_is_masked_anywhere(self):
        from agent import secrets
        secrets.register(["supersecretvalue123"])
        self.assertNotIn("supersecretvalue123", secrets.redact("leak: supersecretvalue123"))

    def test_names_reach_the_system_prompt(self):
        prompt = build_system(default_tools(), r"C:\ws", secret_names=["GH_TOKEN"])
        self.assertIn("GH_TOKEN", prompt)
        self.assertIn("%NAME%", prompt)


class TestAskTool(unittest.TestCase):
    def test_ask_returns_the_question(self):
        tools = {t.name: t for t in default_tools()}
        self.assertIn("ask", tools)
        self.assertEqual(tools["ask"].run(None, question="Which account?"), "Which account?")

    def test_short_answer_rule_present(self):
        self.assertIn("AT MOST 5 lines", build_system(default_tools(), "x"))


class TestPage(unittest.TestCase):
    """Browsing needs structure: an outline to see what is there, absolute links
    to follow, and text in addressable parts so a long page is reachable."""

    HTML = ("<html><head><title>Doc Page</title><style>body{}</style></head><body>"
            "<nav><a href='/home'>Home</a></nav><script>var x=1;</script>"
            "<main><h1>Install</h1><p>Run the installer.</p>"
            "<h2>Options</h2><ul><li>fast</li><li>slow</li></ul>"
            "<p>See <a href='/guide/next'>the next guide</a> and "
            "<a href='https://other.example/x'>another site</a>.</p>"
            "<p>" + ("filler text. " * 500) + "</p></main>"
            "<footer>copyright</footer></body></html>")

    def page(self):
        from agent.page import Page
        return Page(self.HTML, "https://docs.example/start")

    def test_title_and_outline(self):
        page = self.page()
        self.assertEqual(page.title, "Doc Page")
        self.assertIn((1, "Install"), page.headings)
        self.assertIn((2, "Options"), page.headings)

    def test_chrome_is_dropped(self):
        body = self.page().body
        for junk in ("var x=1", "copyright", "body{}"):
            self.assertNotIn(junk, body)

    def test_relative_links_become_absolute(self):
        hrefs = [href for _, href in self.page().links]
        self.assertIn("https://docs.example/guide/next", hrefs)
        self.assertIn("https://other.example/x", hrefs)

    def test_long_text_is_split_into_parts(self):
        page = self.page()
        self.assertGreater(len(page.parts), 1)
        self.assertIn("[part 1/%d]" % len(page.parts), page.part(1))

    def test_part_number_is_clamped(self):
        page = self.page()
        self.assertIn("[part %d/%d]" % (len(page.parts), len(page.parts)), page.part(999))

    def test_outline_mentions_how_to_read_on(self):
        outline = self.page().outline()
        self.assertIn("OUTLINE", outline)
        self.assertIn("mode=text", outline)

    def test_search_reports_the_part(self):
        self.assertIn("[part 1]", self.page().search("Run the installer"))

    def test_search_reports_a_miss(self):
        self.assertIn("no line matched", self.page().search("zzz_absent"))

    def test_search_rejects_a_bad_regex(self):
        self.assertTrue(self.page().search("([").startswith("error"))

    def test_link_list(self):
        listing = self.page().link_list()
        self.assertIn("https://other.example/x", listing)


class TestBrowseTool(unittest.TestCase):
    def setUp(self):
        self.tools = {t.name: t for t in default_tools()}

    def test_registered_and_fetch_is_gone(self):
        self.assertIn("browse", self.tools)
        self.assertNotIn("webfetch", self.tools)

    def test_missing_url(self):
        self.assertTrue(self.tools["browse"].run(None).startswith("error"))

    def test_modes_are_described(self):
        description = self.tools["browse"].description
        for mode in ("outline", "text", "links", "find"):
            self.assertIn(mode, description)


class TestSearxngProvider(unittest.TestCase):
    """The only keyless route that survives an agent searching several times in
    a row: a SearXNG on localhost queries ~20 engines at your own pace."""

    def test_parses_a_searxng_payload(self):
        import json as _json
        from agent import web
        payload = _json.dumps({"results": [
            {"title": "First", "url": "https://a.example", "content": "snippet one"},
            {"title": "Second", "url": "https://b.example", "content": ""}]})
        original = web.get
        web.get = lambda url, **kw: (payload, "application/json")
        try:
            results = web.searxng("anything", 5)
        finally:
            web.get = original
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "https://a.example")
        self.assertEqual(results[0]["snippet"], "snippet one")

    def test_honours_the_base_url_override(self):
        from agent import web
        seen = {}
        original = web.get
        web.get = lambda url, **kw: (seen.setdefault("url", url), '{"results": []}')[1:] \
            and ('{"results": []}', "application/json")
        os.environ["MINIAGENT_SEARX"] = "http://example.test:9999"
        try:
            web.searxng("q", 3)
        finally:
            web.get = original
            del os.environ["MINIAGENT_SEARX"]
        self.assertIn("example.test:9999", seen["url"])

    def test_it_is_first_in_the_keyless_chain(self):
        import inspect
        from agent import web
        source = inspect.getsource(web.search)
        self.assertLess(source.index('"searxng"'), source.index('"duckduckgo"'))


class TestTruncatedToolCall(unittest.TestCase):
    """A write once truncated a file to zero bytes: generation was cut off mid
    <arg_value>, the client closed the tag anyway, so the call parsed cleanly
    with its last argument missing and the tool ran on the fragment."""

    def test_closing_tag_is_not_invented_on_truncation(self):
        from agent.llm import LLMClient
        cut = "<tool_call>write<arg_key>path</arg_key><arg_value>a.html</arg_value>" \
              "<arg_key>text</arg_key><arg_value><!DOCTYPE html><html"
        self.assertEqual(LLMClient._close(cut, "length"), cut)

    def test_closing_tag_is_restored_on_a_normal_stop(self):
        from agent.llm import LLMClient
        call = "<tool_call>sh<arg_key>cmd</arg_key><arg_value>dir</arg_value>"
        self.assertTrue(LLMClient._close(call, "stop").endswith("</tool_call>"))

    def test_truncated_call_loses_its_last_argument_when_closed(self):
        cut = ("<tool_call>write<arg_key>path</arg_key><arg_value>a.html</arg_value>"
               "<arg_key>text</arg_key><arg_value><!DOCTYPE html><html</tool_call>")
        name, args = parse_tool_call(cut)
        self.assertEqual(name, "write")
        self.assertNotIn("text", args)      # exactly the silent data loss

    def test_loop_refuses_to_run_a_truncated_call(self):
        from agent.llm import Usage
        from agent.loop import AgentLoop

        class CutOffClient(object):
            def __init__(self):
                self.calls = 0

            def chat(self, messages, on_delta=None, cancel=None):
                self.calls += 1
                text = ("<tool_call>write<arg_key>path</arg_key><arg_value>x.html"
                        "</arg_value><arg_key>text</arg_key><arg_value><html")
                return text, Usage(10, 10, 0, "length", 0.1, 0.1)

        workdir = os.path.join(os.environ.get("TEMP", "."), "miniagent_trunc")
        os.makedirs(workdir, exist_ok=True)
        target = os.path.join(workdir, "x.html")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("<html>original content</html>")

        cfg = Config(workdir=workdir, max_steps=2)
        session = Session(cfg, default_tools())
        loop = AgentLoop(cfg, session, CutOffClient(), default_tools())
        notes = [d["text"] for name, d in loop.run("GOAL") if name == "note"]

        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "<html>original content</html>")
        self.assertTrue(any("cut off" in n for n in notes), notes)


class TestEmptyWriteGuard(unittest.TestCase):
    def setUp(self):
        self.workdir = os.path.join(os.environ.get("TEMP", "."), "miniagent_empty")
        os.makedirs(self.workdir, exist_ok=True)
        self.session = Session(Config(workdir=self.workdir), default_tools())
        self.write = {t.name: t for t in default_tools()}["write"]

    def test_empty_write_over_existing_content_is_refused(self):
        self.write.run(self.session, path="keep.txt", text="important")
        out = self.write.run(self.session, path="keep.txt", text="")
        self.assertTrue(out.startswith("error"))
        with open(os.path.join(self.workdir, "keep.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "important")

    def test_empty_new_file_is_allowed(self):
        path = os.path.join(self.workdir, "fresh.txt")
        if os.path.exists(path):
            os.remove(path)
        self.assertIn("wrote 0 bytes", self.write.run(self.session, path="fresh.txt", text=""))


class TestMalformedCall(unittest.TestCase):
    """A nested call once reached cmd.exe as a command and came back as
    "< was unexpected at this time.", after which the model abandoned the tool."""

    def test_nested_markup_is_detected(self):
        from agent.protocol import malformed
        nested = ("<tool_call>sh<arg_key>cmd</arg_key><arg_value>browse"
                  "<arg_key>mode</arg_key><arg_value>outline</arg_value>"
                  "<arg_key>url</arg_key><arg_value>https://x</arg_value></tool_call>")
        name, args = parse_tool_call(nested)
        self.assertEqual(name, "sh")
        self.assertTrue(malformed(args))

    def test_clean_call_is_not_flagged(self):
        from agent.protocol import malformed
        _, args = parse_tool_call(
            "<tool_call>sh<arg_key>cmd</arg_key><arg_value>dir /b</arg_value></tool_call>")
        self.assertFalse(malformed(args))

    def test_loop_refuses_to_run_it(self):
        from agent.llm import Usage
        from agent.loop import AgentLoop

        class NestedClient(object):
            def chat(self, messages, on_delta=None, cancel=None):
                text = ("<tool_call>sh<arg_key>cmd</arg_key><arg_value>browse"
                        "<arg_key>mode</arg_key><arg_value>outline</arg_value>"
                        "<arg_key>url</arg_key><arg_value>https://x</arg_value></tool_call>")
                return text, Usage(10, 10, 0, "stop", 0.1, 0.1)

        cfg = Config(workdir=os.environ.get("TEMP", "."), max_steps=2)
        session = Session(cfg, default_tools())
        loop = AgentLoop(cfg, session, NestedClient(), default_tools())
        events = list(loop.run("GOAL"))
        tools_run = [d["name"] for name, d in events if name == "tool_result"]
        notes = [d["text"] for name, d in events if name == "note"]
        self.assertEqual(tools_run, [])
        self.assertTrue(any("malformed" in n for n in notes), notes)


class TestSearchTiming(unittest.TestCase):
    """A dead engine used to cost a full 25s timeout, twice, on every search."""

    def test_search_timeout_is_short(self):
        from agent import web
        self.assertLessEqual(web.SEARCH_TIMEOUT, 10)
        self.assertLess(web.SEARCH_TIMEOUT, web.TIMEOUT)

    def test_a_failing_provider_is_put_in_cooldown(self):
        from agent import web

        def boom(query, limit):
            raise RuntimeError("timed out")

        web._cooldown.clear()
        saved = {name: getattr(web, name) for name in
                 ("searxng", "brave_html", "duckduckgo", "duckduckgo_html",
                  "mojeek", "bing")}
        web.searxng = lambda q, n: []          # no results, so the chain moves on
        web.brave_html = boom
        web.duckduckgo = boom
        web.duckduckgo_html = boom
        web.mojeek = boom
        web.bing = boom
        try:
            try:
                web.search("anything at all", 3)
            except Exception:
                pass
            self.assertIn("duckduckgo", web._cooldown)
            self.assertNotIn("searxng", web._cooldown)
        finally:
            for name, fn in saved.items():
                setattr(web, name, fn)
            web._cooldown.clear()

    def test_empty_results_do_not_trigger_cooldown(self):
        from agent import web

        def boom(query, limit):
            raise RuntimeError("blocked")

        web._cooldown.clear()
        saved = {name: getattr(web, name) for name in
                 ("searxng", "brave_html", "duckduckgo", "duckduckgo_html",
                  "mojeek", "bing")}
        web.searxng = lambda q, n: []
        for name in ("brave_html", "duckduckgo", "duckduckgo_html", "mojeek", "bing"):
            setattr(web, name, boom)
        try:
            try:
                web.search("anything at all", 3)
            except Exception:
                pass
            self.assertNotIn("searxng", web._cooldown)
        finally:
            for name, fn in saved.items():
                setattr(web, name, fn)
            web._cooldown.clear()

    def test_cooldown_is_cleared_when_everything_is_suspended(self):
        from agent import web
        web._cooldown.clear()
        original = web.searxng
        web.searxng = lambda q, n: [{"title": "t", "url": "https://u", "snippet": ""}]
        for name in ("brave-html", "duckduckgo", "duckduckgo-html", "mojeek", "bing"):
            web._cooldown[name] = time.time() + 999
        try:
            provider, results = web.search("anything", 3)
            self.assertEqual(provider, "searxng")
            self.assertTrue(results)
        finally:
            web.searxng = original
            web._cooldown.clear()


class TestProgressGuard(unittest.TestCase):
    """Progress is novelty, not variety of tools. An earlier rule warned after six
    calls to the same tool - which is exactly what researching a site looks like -
    and it cut a good run short at step 11 of a walk through gemtree.com."""

    def setUp(self):
        from agent.guards import Progress
        self.progress = Progress()

    def test_browsing_many_different_pages_is_left_alone(self):
        pages = ["info.htm", "program.htm", "library.htm", "email.php",
                 "GUIDE/index.htm", "faq.htm", "news.htm"]
        for i, page in enumerate(pages, 1):
            observation = "http://site.example/%s\nunique text %d" % (page, i)
            warning = self.progress.push(
                "browse", {"url": "http://site.example/" + page, "mode": "text"},
                observation)
            self.assertEqual(warning, "", "warned on page %d (%s)" % (i, page))

    def test_searches_returning_the_same_links_are_caught(self):
        same = "1. Foo\n   https://a.example\n2. Bar\n   https://b.example"
        queries = ["lm studio spark", "spark lm studio support",
                   "lm studio spark x2", "studio spark lm"]
        warnings = [self.progress.push("websearch", {"query": q}, same) for q in queries]
        self.assertEqual(warnings[:3], ["", "", ""])
        self.assertIn("nothing you had not already seen", warnings[3])

    def test_mixed_work_is_left_alone(self):
        steps = [("read", {"path": "a.py"}, "line 1 of a"),
                 ("edit", {"path": "a.py"}, "edited a.py"),
                 ("sh", {"cmd": "python a.py"}, "output 42"),
                 ("read", {"path": "b.py"}, "line 1 of b"),
                 ("sh", {"cmd": "python b.py"}, "output 7")]
        for name, args, observation in steps:
            self.assertEqual(self.progress.push(name, args, observation), "")

    def test_near_identical_arguments_are_still_caught(self):
        for i in range(2):
            self.assertEqual(
                self.progress.push("websearch", {"query": "spark support"},
                                   "fresh %d https://x%d.example" % (i, i)), "")
            self.progress.push("browse", {"url": "https://u%d" % i},
                               "page %d https://y%d.example" % (i, i))
        warning = self.progress.push("websearch", {"query": "support spark"},
                                     "another https://z.example")
        self.assertIn("near-identical", warning)

    def test_word_order_does_not_hide_a_duplicate(self):
        from agent.guards import Progress
        guard = Progress()
        self.assertEqual(guard._shape("websearch", {"query": "spark lm studio"}),
                         guard._shape("websearch", {"query": "studio lm spark"}))


class TestHints(unittest.TestCase):
    """Every case here is a real error string from a run where the model then
    burned steps guessing. One file per tool, so a new trap costs one entry."""

    def setUp(self):
        from agent.hints import Hints
        self.hints = Hints()

    def fired(self, tool, observation):
        return self.hints.match(tool, observation)

    def test_files_are_loaded_per_tool(self):
        counts = self.hints.by_tool()
        self.assertGreater(counts.get("sh", 0), 5)
        for tool in ("read", "edit", "write", "browse", "grep", "websearch", "any"):
            self.assertIn(tool, counts)

    def test_push_to_a_repository_that_does_not_exist(self):
        hint = self.fired("sh", "exit=128\nremote: Repository not found.\nfatal: not found")
        self.assertIn("gh repo create", hint)

    def test_gh_auth_switch_needs_flags(self):
        hint = self.fired("sh", "exit=1\naccepts 0 arg(s), received 1\nUsage: gh auth switch")
        self.assertIn("--user", hint)

    def test_push_protection(self):
        self.assertIn("secret", self.fired("sh", "remote: error: GH013: Repository rule violations"))

    def test_unix_command_on_windows(self):
        hint = self.fired("sh", "'grep' is not recognized as an internal or external command")
        self.assertIn("findstr", hint)

    def test_read_given_a_url(self):
        hint = self.fired("read", r"error: [Errno 22] Invalid argument: 'C:\ws\http://x.com/'")
        self.assertIn("browse", hint)

    def test_edit_anchor_missing(self):
        self.assertIn("character for character", self.fired("edit", "error: 'old' not found in file"))

    def test_browse_404(self):
        self.assertIn("websearch", self.fired("browse", "error: fetch failed (HTTP Error 404)"))

    def test_search_all_providers_down(self):
        self.assertIn("docker", self.fired("websearch", "error: search failed (searxng: no results)"))

    def test_specific_hint_beats_the_catch_all(self):
        hint = self.fired("sh", "exit=1\nunknown flag: --user\n\nUsage:  gh repo list")
        self.assertIn("does not exist in this version", hint)

    def test_clean_output_gets_nothing(self):
        self.assertEqual(self.fired("sh", "exit=0  cwd=C:/work\\nhello"), "")

    def test_a_hint_is_not_repeated_forever(self):
        from agent.hints import MAX_PER_RUN
        observation = "error: 'old' not found in file"
        given = [bool(self.fired("edit", observation)) for _ in range(MAX_PER_RUN + 2)]
        self.assertEqual(given.count(True), MAX_PER_RUN)
        self.hints.reset()
        self.assertTrue(self.fired("edit", observation))

    def test_hints_only_apply_to_their_own_tool(self):
        self.assertEqual(self.fired("browse", "error: 'old' not found in file"), "")

    def test_an_extra_directory_can_be_merged_in(self):
        import json
        import tempfile
        from agent.hints import Hints
        extra = tempfile.mkdtemp()
        with open(os.path.join(extra, "sh.json"), "w", encoding="utf-8") as fh:
            json.dump([{"when": "wobbly widget", "say": "turn the widget off"}], fh)
        os.environ["MINIAGENT_HINTS"] = extra
        try:
            self.assertIn("turn the widget off", Hints().match("sh", "the wobbly widget failed"))
        finally:
            del os.environ["MINIAGENT_HINTS"]

    def test_broken_json_does_not_break_loading(self):
        import tempfile
        from agent.hints import Hints
        broken = tempfile.mkdtemp()
        with open(os.path.join(broken, "sh.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json at all")
        self.assertEqual(len(Hints(directory=broken)), 0)


class _Plain(object):
    """A style that marks up nothing, so tests can assert on structure."""
    def _same(self, t):
        return t
    bold = italic = underline = dim = cyan = grey = red = green = yellow = magenta = blue = _same


class _Marking(_Plain):
    """A style that tags what it was asked to do."""
    def bold(self, t):
        return "<b>%s</b>" % t

    def italic(self, t):
        return "<i>%s</i>" % t

    def underline(self, t):
        return "<u>%s</u>" % t


class TestMarkdown(unittest.TestCase):
    """The model answers in markdown; printing it raw leaves the reader to
    decode **bold**, [text](url) and pipe tables by eye."""

    def render(self, text, style=None, chunk=7):
        from ui.markdown import MarkdownStream
        out = []
        stream = MarkdownStream(style or _Plain(), out.append)
        for i in range(0, len(text), chunk):     # arrives in stream-sized pieces
            stream.feed(text[i:i + chunk])
        stream.close()
        return "".join(out)

    def test_bold_and_italic(self):
        out = self.render("a **strong** and *slanted* word", _Marking())
        self.assertIn("<b>strong</b>", out)
        self.assertIn("<i>slanted</i>", out)

    def test_bold_italic_together(self):
        self.assertIn("<b><i>both</i></b>", self.render("***both***", _Marking()))

    def test_link_shows_text_and_url(self):
        out = self.render("see [the guide](https://example.com/x) now", _Marking())
        self.assertIn("<u>the guide</u>", out)
        self.assertIn("https://example.com/x", out)

    def test_bold_link(self):
        out = self.render("**[the guide](https://example.com/x)**", _Marking())
        self.assertIn("<u>the guide</u>", out)
        self.assertIn("https://example.com/x", out)

    def test_bare_url_is_marked(self):
        self.assertIn("<u>https://a.example/b</u>",
                      self.render("go to https://a.example/b ok", _Marking()))

    def test_code_span_is_not_treated_as_markup(self):
        out = self.render("`a *b* c`", _Marking())
        self.assertIn("a *b* c", out)
        self.assertNotIn("<i>b</i>", out)

    def test_heading(self):
        self.assertIn("<b>Key Notes</b>", self.render("## Key Notes", _Marking()))

    def test_bullets_get_a_marker(self):
        out = self.render("- one\n- two\n")
        self.assertIn("• one", out)
        self.assertIn("• two", out)

    def test_numbered_list_keeps_its_numbers(self):
        out = self.render("1. first\n2. second\n")
        self.assertIn("1. first", out)
        self.assertIn("2. second", out)

    def test_table_is_aligned_and_separator_row_is_dropped(self):
        table = "| GPU | VRAM |\n|---|---|\n| RTX 3090 | 24 GB |\n| A100 | 80 GB |\n"
        out = self.render(table)
        self.assertNotIn("---", out)
        self.assertIn("│", out)
        rows = [line for line in out.splitlines() if "RTX 3090" in line or "A100" in line]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(rows[1]), "columns are not aligned")

    def test_table_survives_being_split_across_chunks(self):
        table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        self.assertEqual(self.render(table, chunk=3), self.render(table, chunk=200))

    def test_code_fence_contents_are_not_processed(self):
        out = self.render("```python\n**not bold**\n```\n", _Marking())
        self.assertIn("**not bold**", out)
        self.assertNotIn("<b>", out)

    def test_blockquote_and_rule(self):
        out = self.render("> quoted\n\n---\n")
        self.assertIn("quoted", out)
        self.assertIn("─", out)

    def test_plain_text_passes_through(self):
        self.assertIn("just a sentence", self.render("just a sentence"))

    def test_visible_length_ignores_escapes(self):
        from ui.markdown import visible_len
        self.assertEqual(visible_len("\x1b[1mabc\x1b[0m"), 3)


class TestParallelWebTools(unittest.TestCase):
    """Four searches cost 11.7s one at a time and 1.2s together, and the model
    spends another 3-4s generating each separate call."""

    def setUp(self):
        self.tools = {t.name: t for t in default_tools()}

    def test_queries_are_split_one_per_line(self):
        tool = self.tools["websearch"]
        self.assertEqual(tool.targets("a\n b \n\nc"), ["a", "b", "c"])
        self.assertEqual(tool.targets(""), [])

    def test_targets_are_capped(self):
        from agent.tools.web import MAX_TARGETS
        many = "\n".join("q%d" % i for i in range(20))
        self.assertEqual(len(self.tools["websearch"].targets(many)), MAX_TARGETS)

    def test_summary_shows_how_many(self):
        summary = self.tools["websearch"].summary({"query": "one\ntwo\nthree"})
        self.assertIn("one", summary)
        self.assertIn("+2 more", summary)

    def test_parallel_runs_everything_and_keeps_order(self):
        from agent.jobs import parallel
        self.assertEqual(parallel([1, 2, 3], lambda n: n * 10), [10, 20, 30])

    def test_parallel_is_actually_concurrent(self):
        from agent.jobs import parallel
        started = time.time()
        parallel([0.3] * 4, time.sleep)
        self.assertLess(time.time() - started, 0.8)

    def test_a_failing_item_does_not_sink_the_rest(self):
        from agent.jobs import parallel

        def flaky(n):
            if n == 2:
                raise ValueError("boom")
            return "ok %d" % n
        results = parallel([1, 2, 3], flaky)
        self.assertEqual(results[0], "ok 1")
        self.assertTrue(results[1].startswith("error:"))
        self.assertEqual(results[2], "ok 3")


class TestBackgroundJobs(unittest.TestCase):
    def setUp(self):
        from agent.jobs import JobRunner
        self.runner = JobRunner()

    def tearDown(self):
        self.runner.shutdown()

    def test_submit_returns_at_once(self):
        started = time.time()
        job = self.runner.submit("slow thing", lambda: time.sleep(0.4) or "done")
        self.assertLess(time.time() - started, 0.2)
        self.assertTrue(job.startswith("job"))
        self.assertEqual(self.runner.busy(), 1)

    def test_results_are_drained_when_ready(self):
        self.runner.submit("thing", lambda: "the answer")
        finished = self.runner.drain(wait=2.0)
        self.assertEqual(len(finished), 1)
        job, label, result, seconds = finished[0]
        self.assertEqual(label, "thing")
        self.assertEqual(result, "the answer")
        self.assertEqual(self.runner.busy(), 0)

    def test_drain_without_waiting_returns_nothing_yet(self):
        self.runner.submit("slow", lambda: time.sleep(1.0) or "late")
        self.assertEqual(self.runner.drain(), [])
        self.assertEqual(self.runner.busy(), 1)

    def test_a_raising_job_comes_back_as_an_error(self):
        self.runner.submit("bad", lambda: (_ for _ in ()).throw(ValueError("nope")))
        finished = self.runner.drain(wait=2.0)
        self.assertIn("error:", finished[0][2])

    def test_background_flag_hands_work_to_the_runner(self):
        class Ctx(object):
            pass
        ctx = Ctx()
        ctx.jobs = self.runner
        tool = {t.name: t for t in default_tools()}["websearch"]
        out = tool.run(ctx, query="anything", background="true")
        self.assertIn("background", out)
        self.assertIn("job", out)

    def test_without_a_runner_it_just_runs_normally(self):
        tool = {t.name: t for t in default_tools()}["websearch"]
        self.assertIsNone(tool.background(object(), {"background": "true"}, "l", lambda: "x"))


class TestRepeatedArgumentKeys(unittest.TestCase):
    """The model says "search for these four things" as four <arg_key>query</arg_key>
    pairs. Folding those into a dict kept the last one and dropped three."""

    def test_repeated_key_becomes_several_lines(self):
        call = ("<tool_call>websearch"
                "<arg_key>query</arg_key><arg_value>one</arg_value>"
                "<arg_key>query</arg_key><arg_value>two</arg_value>"
                "<arg_key>query</arg_key><arg_value>three</arg_value></tool_call>")
        name, args = parse_tool_call(call)
        self.assertEqual(name, "websearch")
        self.assertEqual(args["query"].splitlines(), ["one", "two", "three"])

    def test_single_key_is_unchanged(self):
        self.assertEqual(
            parse_tool_call("<tool_call>sh<arg_key>cmd</arg_key>"
                            "<arg_value>dir /b</arg_value></tool_call>"),
            ("sh", {"cmd": "dir /b"}))

    def test_distinct_keys_are_unaffected(self):
        _, args = parse_tool_call(
            "<tool_call>edit<arg_key>path</arg_key><arg_value>a.py</arg_value>"
            "<arg_key>old</arg_key><arg_value>x</arg_value>"
            "<arg_key>new</arg_key><arg_value>y</arg_value></tool_call>")
        self.assertEqual(args, {"path": "a.py", "old": "x", "new": "y"})

    def test_the_web_tool_then_sees_every_target(self):
        call = ("<tool_call>browse"
                "<arg_key>url</arg_key><arg_value>https://a.example</arg_value>"
                "<arg_key>url</arg_key><arg_value>https://b.example</arg_value></tool_call>")
        _, args = parse_tool_call(call)
        tool = {t.name: t for t in default_tools()}["browse"]
        self.assertEqual(tool.targets(args["url"]),
                         ["https://a.example", "https://b.example"])


class TestAsyncWebSwitch(unittest.TestCase):
    """--async-web forces every web call into the background. Measured on one
    task it was worse in every way, so it is off by default and kept only so the
    experiment can be repeated."""

    def setUp(self):
        from agent.jobs import JobRunner
        self.runner = JobRunner()
        self.tool = {t.name: t for t in default_tools()}["websearch"]

    def tearDown(self):
        self.runner.shutdown()

    def ctx(self, async_web):
        holder = type("Ctx", (), {})()
        holder.jobs = self.runner
        holder.cfg = Config(async_web=async_web)
        return holder

    def test_off_by_default(self):
        self.assertFalse(Config().async_web)

    def test_flag_forces_background_without_the_model_asking(self):
        out = self.tool.run(self.ctx(True), query="anything")
        self.assertIn("background", out)

    def test_without_the_flag_an_ordinary_call_still_blocks(self):
        self.assertIsNone(
            self.tool.background(self.ctx(False), {}, "label", lambda: "x"))
