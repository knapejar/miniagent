# -*- coding: utf-8 -*-
"""e01: a /version command in miniagent.py, the suite still green."""
import contextlib
import io
import re

from common import RESULTS, check, load, report, source, suite_passes

text = source("miniagent.py")
check("compile", "miniagent.py compiles", lambda: bool(compile(text, "miniagent.py", "exec")))
app = load("miniagent")


def run_command(line):
    from agent.config import Config
    from agent.metrics import Metrics
    from agent.session import Session
    from agent.tools import default_tools
    from agent.trace import Trace
    from ui.render import Renderer
    cfg = Config(workdir=".", profile=None, color=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        app.handle_command(line, cfg, Session(cfg, default_tools()), Renderer(cfg, Metrics()),
                           Trace(""), None)
    return out.getvalue()


help_lines = app.HELP.splitlines()
at = [i for i, l in enumerate(help_lines) if l.strip().startswith("/help")]
check("help", "HELP lists /version", lambda: "/version" in app.HELP)
check("help", "/version is the line right below /help",
      lambda: bool(at) and help_lines[at[0] + 1].strip().startswith("/version"))
check("command", "/version prints VERSION", lambda: app.VERSION in run_command("/version"))
check("command", "/version is not 'unknown command'",
      lambda: "unknown command" not in run_command("/version"))
check("command", "/trace still works", lambda: "unknown command" not in run_command("/trace"))


def near_trace():
    body = text[text.index("def handle_command"):]
    lines = body.splitlines()
    version = [i for i, l in enumerate(lines) if re.search(r"[\"']/version[\"']", l)]
    trace = [i for i, l in enumerate(lines) if re.search(r"[\"']/trace[\"']", l)]
    return bool(version and trace) and abs(version[0] - trace[0]) <= 6


check("command", "the handler sits next to /trace", near_trace)
suite_passes()
report()
