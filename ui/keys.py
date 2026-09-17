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


class CancelWatcher(object):
    """Sets `event` when the user presses ESC and `background` on ctrl+b (move the
    command the agent is waiting for to the background). No-op when stdin is not a TTY."""

    def __init__(self, event, background=None):
        self.event = event
        self.background = background
        self._stop = threading.Event()
        self._thread = None
        self._restore = None

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
