# -*- coding: utf-8 -*-
"""Run commands, in cmd.exe or PowerShell.

The model constantly mixes the two shells: it writes `$env:X = "y"; cmd`,
`Get-Process | Where-Object {...}` or `| Select-String` and then sends it to
cmd.exe, which fails with a syntax error it cannot diagnose. Detecting the
dialect and routing to the right interpreter removes an entire class of
failures that no amount of prompting fixed.
"""
import os
import re
import subprocess
import tempfile
import threading
import time

# Commands that always require confirmation in approve=auto mode.
RISKY = re.compile(
    r"(?i)\b(format|diskpart|bcdedit|shutdown|takeown|icacls|netsh|vssadmin|cipher"
    r"|wmic|reg\s+(add|delete)|r?mdir\s+/s|del\s+/[sqf]|schtasks|sc\s+(create|delete)"
    r"|net\s+user|curl|wget|pip\s+install|npm\s+i|auth\s+token)\b")

# Syntax that only PowerShell understands.
POWERSHELL_HINTS = re.compile(
    r"(\$env:|\$_|\$\(|\bGet-\w+|\bSet-\w+|\bNew-\w+|\bTest-Path\b|\bWrite-Host\b"
    r"|\bSelect-(Object|String)\b|\bWhere-Object\b|\bForEach-Object\b|\bOut-File\b"
    r"|\bMeasure-Object\b|\bConvertFrom-\w+|\bStart-Process\b|-ErrorAction\b)")

# Seconds of runtime at which the model has to decide about a command that is still
# running: wait on, move it to the background or kill it. The last is a final
# warning - a command still running at STALL is killed.
CHECKPOINTS = (1, 30, 180, 300)
STALL = 600

CMD_SCRIPT = ('@echo off\r\n'
              'chcp 65001>nul\r\n'
              'cd /d "%s"\r\n'
              '%s\r\n'
              'set __rc=%%errorlevel%%\r\n'
              '>"%s" (echo %%__rc%%& cd)\r\n'
              'exit /b %%__rc%%\r\n')

PS_SCRIPT = ('$ErrorActionPreference = "Continue"\n'
             '[Console]::OutputEncoding = [Text.Encoding]::UTF8\n'
             'Set-Location -LiteralPath %s\n'
             '%s\n'
             '$__rc = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } '
             'elseif ($?) { 0 } else { 1 }\n'
             'Set-Content -LiteralPath %s -Value @($__rc, (Get-Location).Path) '
             '-Encoding utf8\n'
             'exit $__rc\n')


def looks_like_powershell(cmd):
    """True when the command uses syntax cmd.exe cannot parse."""
    if re.match(r"(?i)^\s*(powershell|pwsh)\b", cmd):
        return False           # already an explicit invocation, let cmd run it
    return bool(POWERSHELL_HINTS.search(cmd))


class ShellTool(object):
    name = "sh"
    description = ("Run a command. cmd.exe by default; PowerShell syntax is routed there "
                   "automatically. Multi-line input runs as a script, the working directory "
                   "persists, and %% means a literal percent in cmd.")
    params = {"cmd": {"type": "string"},
              "shell": {"type": "string", "description": "cmd or powershell, default: auto"}}
    required = ("cmd",)
    danger = True

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def summary(self, args):
        shell = self._pick(args.get("cmd", ""), args.get("shell"))
        prefix = "" if shell == "cmd" else "[ps] "
        return prefix + str(args.get("cmd", "")).replace("\n", " ; ")[:100]

    def needs_approval(self, args):
        return bool(RISKY.search(args.get("cmd", "")))

    @staticmethod
    def _pick(cmd, requested):
        if requested:
            return "powershell" if str(requested).lower().startswith(("p", "ps")) else "cmd"
        return "powershell" if looks_like_powershell(cmd) else "cmd"

    def run(self, ctx, cmd=None, shell=None, **_):
        if cmd is None:
            return "error: missing arg 'cmd'"
        dialect = self._pick(cmd, shell)
        tmp = tempfile.mkdtemp(prefix="miniagent_")
        state = os.path.join(tmp, "state.txt")

        if dialect == "powershell":
            script_path = os.path.join(tmp, "run.ps1")
            body = PS_SCRIPT % (_ps_quote(ctx.cwd), cmd, _ps_quote(state))
            argv = ["powershell", "-NoProfile", "-NonInteractive",
                    "-ExecutionPolicy", "Bypass", "-File", script_path]
            newline = "\n"
        else:
            script_path = os.path.join(tmp, "run.cmd")
            # The batch body already carries \r\n, so newline="" keeps it verbatim.
            body = CMD_SCRIPT % (ctx.cwd, cmd.replace("\n", "\r\n"), state)
            argv = ["cmd.exe", "/d", "/s", "/c", script_path]
            newline = ""
        with open(script_path, "w", encoding="utf-8", newline=newline) as fh:
            fh.write(body)

        output = os.path.join(tmp, "output.txt")
        with open(output, "wb") as sink:
            try:
                proc = subprocess.Popen(argv, stdout=sink, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, cwd=ctx.cwd)
            except FileNotFoundError as e:
                return "error: cannot start %s (%s)" % (dialect, e)
        job = Running(_registry(ctx), cmd, dialect, proc, output, state)
        return wait(ctx, job, CHECKPOINTS[0])

    def run_foreground(self, ctx, cmd):
        """For `!command` typed by the user: no checkpoints, just the result."""
        out = self.run(ctx, cmd=cmd)
        for job in list(_registry(ctx).values()):
            if job.undecided:
                job.proc.wait()
                return finish(ctx, job)
        return out

    @staticmethod
    def _adopt_cwd(ctx, state):
        if not os.path.exists(state):
            return
        try:
            with open(state, encoding="utf-8-sig", errors="replace") as fh:
                lines = [l.strip() for l in fh.read().splitlines() if l.strip()]
            if len(lines) > 1 and os.path.isdir(lines[1]):
                ctx.cwd = lines[1]
        except Exception:
            pass


class Running(object):
    """A command that outlived the first checkpoint."""

    def __init__(self, registry, cmd, dialect, proc, output, state):
        n = len(registry) + 1
        while "sh%d" % n in registry:
            n += 1
        self.id = "sh%d" % n
        self.cmd, self.dialect, self.proc = cmd, dialect, proc
        self.output, self.state = output, state
        self.started = time.time()
        self.undecided = False        # the model was asked and has not answered yet
        self.lock = threading.Lock()
        self.done = None              # the formatted result, once read
        registry[self.id] = self

    def elapsed(self):
        return time.time() - self.started

    def tail(self, lines=5):
        try:
            with open(self.output, "rb") as fh:
                fh.seek(max(0, os.path.getsize(self.output) - 2000))
                text = fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return "\n".join(text.strip().splitlines()[-lines:])

    def label(self):
        return "sh %s: %s" % (self.id, self.cmd.replace("\n", " ; ")[:80])


def _registry(ctx):
    registry = getattr(ctx, "shell_jobs", None)
    if registry is None:
        registry = ctx.shell_jobs = {}
    return registry


def _event(ctx, name):
    event = getattr(ctx, name, None)
    return event if isinstance(event, threading.Event) else None


def next_checkpoint(job):
    for mark in CHECKPOINTS:
        if mark > job.elapsed():
            return mark
    return STALL


def wait(ctx, job, until):
    """Block until the command ends, its runtime reaches `until` seconds, the
    user presses ctrl+b (background) or esc (stop the run)."""
    job.undecided = False
    background, cancel = _event(ctx, "background_key"), _event(ctx, "cancel")
    if background is not None:
        background.clear()                # a press from before this wait does not count
    while job.proc.poll() is None:
        if background is not None and background.is_set():
            background.clear()
            return to_background(ctx, job, "the user pressed ctrl+b")
        if cancel is not None and cancel.is_set():
            kill(ctx, job)
            return "killed %s after %.0fs: the user stopped the run" % (job.id, job.elapsed())
        if job.elapsed() >= until:
            if until >= STALL:
                kill(ctx, job)
                return ("error: %s stalled and was killed after %ds: %s\nlast output:\n%s\n"
                        "Find a faster approach."
                        % (job.id, STALL, job.cmd[:200], job.tail() or "(none)"))
            return decision_prompt(job)
        time.sleep(0.05)
    return finish(ctx, job)


def decision_prompt(job):
    job.undecided = True
    upcoming = next_checkpoint(job)
    if upcoming >= STALL:
        head = ("[LAST WARNING] %s is still running after %.0fs. If you wait again, it is "
                "killed as stalled at %ds." % (job.id, job.elapsed(), STALL))
        then = "killed at %ds if still running" % STALL
    else:
        head = "[RUNNING] %s is still running after %.0fs." % (job.id, job.elapsed())
        then = "asked again at %ds" % upcoming
    return ("%s\ncommand: %s\nlast output:\n%s\n"
            "Decide now with the wait tool (job=%s):\n"
            "  action=wait        wait for it to end, %s\n"
            "  action=background  keep working; its result is delivered to you when it ends\n"
            "  action=kill        stop it and find a faster approach\n"
            "Calling any other tool moves it to the background."
            % (head, job.cmd.replace("\n", " ; ")[:200], job.tail() or "(none yet)",
               job.id, then))


def finish(ctx, job):
    """Read the result of an ended command (once) and forget the job."""
    with job.lock:
        if job.done is None:
            job.proc.wait()
            ShellTool._adopt_cwd(ctx, job.state)
            try:
                with open(job.output, "rb") as fh:
                    raw = fh.read()
            except OSError:
                raw = b""
            out = raw.decode("utf-8", "replace").strip() or "(no output)"
            label = "" if job.dialect == "cmd" else "  shell=powershell"
            job.done = "exit=%d  (%.1fs)%s  cwd=%s\n%s" % (
                job.proc.returncode, job.elapsed(), label, ctx.cwd, out)
        _registry(ctx).pop(job.id, None)
        return job.done


def to_background(ctx, job, why):
    job.undecided = False
    runner = getattr(ctx, "jobs", None)
    if runner is None:
        return wait(ctx, job, STALL)
    runner.submit(job.label(), lambda: finish(ctx, job))
    return ("moved %s to the background after %.0fs (%s). Keep working - its result is "
            "delivered to you automatically when it ends." % (job.id, job.elapsed(), why))


def kill(ctx, job):
    job.undecided = False
    if job.proc.poll() is None:
        if os.name == "nt":               # cmd.exe/powershell would leave children running
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(job.proc.pid)],
                           capture_output=True)
        try:
            job.proc.kill()
        except OSError:
            pass
    job.proc.wait()
    _registry(ctx).pop(job.id, None)


def auto_background(ctx):
    """The model moved on without deciding: background what it was asked about."""
    return [to_background(ctx, job, "no decision was made")
            for job in list(_registry(ctx).values()) if job.undecided]


def kill_all(ctx):
    for job in list(_registry(ctx).values()):
        kill(ctx, job)


class WaitTool(object):
    name = "wait"
    description = "Decide about a running sh job: action=wait|background|kill."
    params = {"job": {"type": "string"}, "action": {"type": "string"}}
    required = ("job", "action")
    danger = False

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return False

    @staticmethod
    def _action(value):
        value = str(value or "").strip().lower()
        for action in ("wait", "background", "kill"):
            if value[:2] == action[:2]:
                return action
        return None

    def summary(self, args):
        action = self._action(args.get("action")) or "?"
        return "%s -> %s" % (args.get("job", "?"), action.upper())

    def run(self, ctx, job=None, action=None, **_):
        registry = _registry(ctx)
        running = registry.get(str(job or "").strip())
        if running is None:
            return "error: no running command '%s'. Running: %s" % (
                job, ", ".join(sorted(registry)) or "none")
        action = self._action(action)
        if action is None:
            return "error: action must be wait, background or kill"
        if action == "kill":
            tail = running.tail()
            kill(ctx, running)
            return "killed %s after %.0fs\nlast output:\n%s" % (
                running.id, running.elapsed(), tail or "(none)")
        if action == "background":
            return to_background(ctx, running, "the model decided so")
        return wait(ctx, running, next_checkpoint(running))


def _ps_quote(value):
    return "'%s'" % value.replace("'", "''")
