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

TIMEOUT = 120

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

        started = time.time()
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=TIMEOUT, cwd=ctx.cwd)
            code, raw = proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            code = -1
            raw = (e.stdout or b"") + (e.stderr or b"") + b"\n[killed after timeout]"
        except FileNotFoundError as e:
            return "error: cannot start %s (%s)" % (dialect, e)

        self._adopt_cwd(ctx, state)
        out = raw.decode("utf-8", "replace").strip() or "(no output)"
        label = "" if dialect == "cmd" else "  shell=powershell"
        return "exit=%d  (%.1fs)%s  cwd=%s\n%s" % (
            code, time.time() - started, label, ctx.cwd, out)

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


def _ps_quote(value):
    return "'%s'" % value.replace("'", "''")
