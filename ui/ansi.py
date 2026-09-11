# -*- coding: utf-8 -*-
"""ANSI colours and small terminal helpers. No dependencies."""
import os
import sys

CSI = "\x1b["
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Style(object):
    """Colours can be switched off with a single flag (--no-color)."""

    def __init__(self, enabled=True):
        self.enabled = enabled and _supports_color()

    def _w(self, code, text):
        return "%s%sm%s%s0m" % (CSI, code, text, CSI) if self.enabled else text

    def dim(self, t):
        return self._w("2", t)

    def bold(self, t):
        return self._w("1", t)

    def italic(self, t):
        return self._w("3", t)

    def underline(self, t):
        return self._w("4", t)

    def red(self, t):
        return self._w("31", t)

    def green(self, t):
        return self._w("32", t)

    def yellow(self, t):
        return self._w("33", t)

    def blue(self, t):
        return self._w("34", t)

    def magenta(self, t):
        return self._w("35", t)

    def cyan(self, t):
        return self._w("36", t)

    def grey(self, t):
        return self._w("90", t)


def _supports_color():
    if os.getenv("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        return _enable_windows_vt()
    return True


def _enable_windows_vt():
    """Enable ANSI escape handling in the Windows console (without colorama)."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        mode = ctypes.c_uint32()
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def setup_stdio():
    """The Windows console is cp1252/cp852 - without this, printing a non-ASCII
    character raises UnicodeEncodeError and kills the run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def clear_line():
    return "\r" + CSI + "2K"


def human_tokens(n):
    if n < 1000:
        return str(n)
    if n < 1000000:
        return "%.1fk" % (n / 1000.0)
    return "%.2fM" % (n / 1000000.0)


def human_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def bar(pct, width=10):
    filled = int(round(width * min(100.0, max(0.0, pct)) / 100.0))
    return "█" * filled + "░" * (width - filled)
