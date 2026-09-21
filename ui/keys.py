# -*- coding: utf-8 -*-
"""Watch for a cancel keypress while the agent is working.

The agent loop can run for minutes at a time. Ctrl+C works but is abrupt and
on Windows it can land in the middle of a subprocess. ESC is friendlier: a
background thread polls the keyboard and sets an Event, which the loop and the
streaming client both check, so generation stops mid-token instead of after the
current step.
"""
import os
import sys
import threading

ESC = "\x1b"
CTRL_B = "\x02"
BACKSPACE = ("\x08", "\x7f")


def push_back(text):
    """Hand text the user typed during a run back to the terminal's own line
    editor, as if they had just typed it at the prompt.

    Echoing it in front of input() instead looks the same until you try to
    erase it: the line editor never saw those characters, so backspace stops at
    the prompt and the text cannot be taken back. Returns False when the
    terminal will not take it, and the caller falls back to echoing.
    """
    if not text:
        return True
    try:
        if not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return _push_windows(text) if os.name == "nt" else _push_readline(text)


def _push_readline(text):
    try:
        import readline
    except ImportError:
        return False

    def hook():
        readline.insert_text(text)
        readline.redisplay()
        readline.set_startup_hook(None)     # once, for the next prompt only

    readline.set_startup_hook(hook)
    return True


def input_record_type():
    """The Win32 INPUT_RECORD, built on demand. Its size has to come out at 20
    bytes (16 of them the KEY_EVENT_RECORD) or WriteConsoleInputW reads past the
    end of every record."""
    import ctypes
    from ctypes import wintypes

    class _Char(ctypes.Union):
        _fields_ = [("UnicodeChar", wintypes.WCHAR), ("AsciiChar", ctypes.c_char)]

    class _Key(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", _Char), ("dwControlKeyState", wintypes.DWORD)]

    class _Event(ctypes.Union):
        _fields_ = [("KeyEvent", _Key)]

    class _Record(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", _Event)]

    return _Record


def _push_windows(text):
    """Write the characters into the console's input buffer (WriteConsoleInputW),
    which is the queue the line editor reads from."""
    try:
        import ctypes
        from ctypes import wintypes

        _Record = input_record_type()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetStdHandle(-10)          # STD_INPUT_HANDLE
        if handle in (0, -1, None):
            return False
        records = (_Record * (len(text) * 2))()
        for i, char in enumerate(text):
            for j, down in ((0, 1), (1, 0)):         # each key goes down, then up
                record = records[i * 2 + j]
                record.EventType = 1                 # KEY_EVENT
                record.Event.KeyEvent.bKeyDown = down
                record.Event.KeyEvent.wRepeatCount = 1
                record.Event.KeyEvent.uChar.UnicodeChar = char
        written = wintypes.DWORD(0)
        ok = kernel32.WriteConsoleInputW(handle, records, len(records),
                                         ctypes.byref(written))
        return bool(ok) and written.value == len(records)
    except Exception:
        return False


class CancelWatcher(object):
    """Sets `event` when the user presses ESC and `background` on ctrl+b (move the
    command the agent is waiting for to the background). No-op when stdin is not a TTY."""

    def __init__(self, event, background=None):
        self.event = event
        self.background = background
        self._stop = threading.Event()
        self._thread = None
        self._restore = None
        self._typed = []

    @property
    def active(self):
        return self._thread is not None

    def start(self):
        if not sys.stdin.isatty():
            return self
        reader = _windows_reader() if os.name == "nt" else self._posix_reader()
        if reader is None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(reader,), daemon=True)
        self._thread.start()
        return self

    def _run(self, reader):
        while not self._stop.is_set():
            key = reader(0.1)
            if key == ESC:
                self.event.set()
                return
            if key == CTRL_B and self.background is not None:
                self.background.set()
            elif key in BACKSPACE:
                if self._typed:
                    self._typed.pop()
            elif key and key >= " " and key != "\x7f":
                # Anything else the user types while the agent works is kept, not
                # eaten: the REPL puts it back in front of the next prompt.
                self._typed.append(key)

    def typed(self):
        """What the user typed while the agent was working, and forget it."""
        text = "".join(self._typed).strip()
        self._typed = []
        return text

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None
        if self._restore:
            self._restore()
            self._restore = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------- platforms
    def _posix_reader(self):
        try:
            import select
            import termios
            import tty
        except ImportError:
            return None
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)

        def read(timeout):
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            return sys.stdin.read(1) if ready else None
        return read


def _windows_reader():
    try:
        import msvcrt
    except ImportError:
        return None

    def read(timeout):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if msvcrt.kbhit():
                try:
                    return msvcrt.getwch()
                except Exception:
                    return None
            time.sleep(0.02)
        return None
    return read
