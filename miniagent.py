#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""miniagent - an agentic CLI for local models.

    python miniagent.py                     interactive
    python miniagent.py "task"              one shot
    python miniagent.py -f tasks/x.txt      task from a file
"""
import sys
import threading
import os

from agent.config import EFFORTS, parse_args
from agent.llm import LLMClient
from agent.loop import AgentLoop
from agent.metrics import Metrics
from agent.session import Session
from agent.tools import default_tools
from agent.tools.shell import kill_all
from agent.trace import Trace
from ui import ansi
from ui.keys import CancelWatcher
from ui.render import Renderer

VERSION = "0.2.0"

HELP = """  commands
    /help              this help
    /plan              show the agent's pinned plan
    /stats             metrics of the last run
    /tools             tools and their schemas
    /hints [tool]      hints loaded for each tool (hints/*.json)
    /context           what is currently taking up the context
    /history           the whole conversation, model answers and tool results
    /cwd [path]        show or change the working directory
    /effort <level>    default | none | low | medium | high | xhigh
    /kaggle            re-check the Kaggle model and start its proxy (--kaggle)
    /approve <mode>    auto | all | none
    /steps <n>         step limit per task
    /clear [all]       drop the conversation; the plan survives unless 'all'
    /edit <text>       replace the current task (e.g. after cancelling it)
    /trace             path to the JSONL trace
    /quit              exit
    !<command>         run a shell command directly, without the model

  esc during a run stops the agent immediately
  ctrl+b moves a command the agent is waiting for to the background
"""


def run_task(cfg, session, client, tools, trace, renderer, task):
    """Handle one task. Returns the metrics snapshot."""
    cancel = threading.Event()
    loop = AgentLoop(cfg, session, client, tools, trace,
                     approve_fn=renderer.approval, cancel=cancel)
    renderer.metrics = loop.metrics
    snapshot = {}
    watcher = CancelWatcher(cancel, session.background_key).start()
    try:
        for event, data in loop.run(task, on_delta=renderer.on_delta):
            renderer.handle(event, data)
            if event == "end":
                snapshot = data.get("metrics") or {}
    except KeyboardInterrupt:
        renderer._stop_spinner()
        print("\n  " + renderer.s.yellow("interrupted"))
        snapshot = loop.metrics.snapshot()
    except Exception as e:                  # a failed task must never close the session
        renderer._stop_spinner()
        print("\n  " + renderer.s.red("error: %s: %s" % (type(e).__name__, e)))
        if trace:
            trace.write("crash", err="%s: %s" % (type(e).__name__, e))
        snapshot = loop.metrics.snapshot()
    finally:
        watcher.stop()
    return snapshot


def connect_kaggle(cfg, renderer):
    """Point cfg at Qwen3.8 on the Kaggle TPU (agent/kaggle.py)."""
    from agent import kaggle
    return kaggle.setup(cfg, say=lambda text: print("  " + renderer.s.dim(text)))


def handle_command(line, cfg, session, renderer, trace, last):
    """Return False when the REPL should exit."""
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    s = renderer.s

    if cmd in ("/quit", "/exit", "/q"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/plan":
        if session.plan_text:
            print("\n  " + s.bold("agent plan") +
                  s.dim(" (%d chars, pinned in context)" % len(session.plan_text)))
            for entry in session.plan_text.splitlines():
                print("    " + entry)
            print()
        else:
            print("  " + s.dim("the plan is empty - the agent writes it with the plan tool"))
    elif cmd == "/stats":
        if last:
            renderer.summary(last)
        else:
            print("  " + s.dim("no run yet"))
    elif cmd == "/tools":
        for tool in session.tools:
            print("  " + s.bold(tool.name) + s.dim("(" + ", ".join(tool.params) + ")"))
            print("    " + s.dim(tool.description))
    elif cmd == "/hints":
        from agent.hints import Hints
        hints = Hints()
        if arg:
            rows = [e for e in hints.entries if e["tool"] == arg]
            print("  " + s.bold("%s" % arg) + s.dim("  %d hints" % len(rows)))
            for entry in rows:
                print("    " + s.dim(entry["when"][:40].ljust(42)) + entry["say"][:90])
        else:
            print("  " + s.dim("%d hints in hints/*.json - add one to a tool's file to "
                               "teach it a new recovery" % len(hints)))
            for tool, count in sorted(hints.by_tool().items()):
                print("    %-12s %d" % (tool, count))
    elif cmd == "/context":
        total = session.total_chars()
        print("  " + s.dim("messages: %d   chars: %d   estimate: %d tokens (%.2f chars/tok)"
                           % (len(session.messages), total, session.estimated_tokens(),
                              session.chars_per_token)))
        labels = ["system", "task", "plan"]
        for i, message in enumerate(session.messages[:3]):
            label = labels[i] if i < len(labels) else message["role"]
            print("    [%d] %-8s %6d chars" % (i, label, len(message.get("content") or "")))
        rest = sum(len(m.get("content") or "") for m in session.messages[3:])
        print("    [3:] %-8s %6d chars  (%d messages, this is what gets cropped)"
              % ("run", rest, max(0, len(session.messages) - 3)))
    elif cmd == "/history":
        if len(session.messages) <= 1:
            print("  " + s.dim("no conversation yet"))
        for i, message in enumerate(session.messages[1:], 1):
            content = (message.get("content") or "").strip()
            if len(content) > 500:
                content = content[:500] + " ..."
            print("  " + s.bold("[%d] %s" % (i, message["role"])))
            thinking = (message.get("reasoning_content") or "").strip()
            if thinking:
                if len(thinking) > 500:
                    thinking = thinking[:500] + " ..."
                print("    " + s.grey("thinking:"))
                for entry in thinking.splitlines():
                    print("      " + s.grey(entry))
            for entry in content.splitlines():
                print("    " + entry)
    elif cmd == "/edit":
        if not arg:
            print("  " + s.dim("/edit <text> replaces the current task"))
        elif len(session.messages) < 2:
            print("  " + s.dim("no task yet - just type it"))
        else:
            session.messages[1] = {"role": "user", "content": arg}
            session.goal = arg
            print("  " + s.dim("task replaced (%d chars)" % len(arg)))
    elif cmd == "/cwd":
        if arg:
            import os
            path = os.path.abspath(arg)
            if os.path.isdir(path):
                session.cwd = path
            else:
                print("  " + s.red("not a directory: " + path))
        print("  " + session.cwd)
    elif cmd == "/effort":
        if arg in EFFORTS:
            cfg.effort = arg
        print("  " + s.dim("reasoning: %s" % cfg.effort))
    elif cmd == "/approve":
        if arg in ("auto", "all", "none"):
            cfg.approve = arg
        print("  " + s.dim("approval: " + cfg.approve))
    elif cmd == "/steps":
        if arg.isdigit():
            cfg.max_steps = int(arg)
        print("  " + s.dim("step limit: %d" % cfg.max_steps))
    elif cmd == "/clear":
        dropped = session.reset(keep_plan=arg != "all")
        kept = "the plan is kept" if session.plan_text else "nothing kept"
        print("  " + s.dim("dropped %d messages; %s (/clear all drops the plan too)"
                           % (dropped, kept)))
    elif cmd == "/kaggle":
        if cfg.profile != "kaggle":
            print("  " + s.dim("start miniagent with --kaggle to use the Kaggle model"))
        elif connect_kaggle(cfg, renderer):
            print("  " + s.green("ready") + s.dim("  %s   context %d   @ %s"
                                                  % (cfg.model, cfg.ctx, cfg.url)))
    elif cmd == "/trace":
        print("  " + (trace.path or "tracing is off"))
    else:
        print("  " + s.dim("unknown command, try /help"))
    return True


def main(argv=None):
    ansi.setup_stdio()
    cfg, task = parse_args(argv)

    if cfg.profile == "kaggle":
        try:
            connect_kaggle(cfg, Renderer(cfg, Metrics()))
        except KeyboardInterrupt:
            print()
            return 130
    else:
        from agent import lmstudio
        say = Renderer(cfg, Metrics()).s.dim
        lmstudio.setup(cfg, say=lambda text: print("  " + say(text)))

    tools = default_tools()
    session = Session(cfg, tools)
    client = LLMClient(cfg)
    trace = Trace(cfg.trace_dir, cfg.run_name)
    renderer = Renderer(cfg, Metrics())

    # Make sure the configured working directory is active before starting
    # a run, so the agent and any tools operate where the user asked.
    os.chdir(cfg.workdir)

    if task:                                   # one-shot run
        snapshot = run_task(cfg, session, client, tools, trace, renderer, task)
        renderer.summary(snapshot)
        kill_all(session)
        trace.close()
        return 0

    renderer.banner(session)
    last = None
    while True:
        try:
            line = input(renderer.s.cyan("› ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, cfg, session, renderer, trace, last):
                break
            continue
        if line.startswith("!"):
            from agent.tools.shell import ShellTool
            print(renderer.s.grey(ShellTool().run_foreground(session, line[1:])))
            continue
        last = run_task(cfg, session, client, tools, trace, renderer, line)

    kill_all(session)
    trace.close()
    print(renderer.s.dim("  bye"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
