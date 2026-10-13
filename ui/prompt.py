# -*- coding: utf-8 -*-
"""The prompt's own line editor.

input() takes a multi-line paste one line at a time: the first line runs as a
task and every other line queues up as a task of its own. Here a paste stays
on the line as one [Pasted text #1 +12 lines], a single backspace takes it out
again, and the text itself goes in its place when the line is sent.
"""
import codecs
import contextlib
import itertools
import os
import re
import shutil
import sys
import time
import unicodedata

KEYS = {"\x1b[D": "left", "\x1bOD": "left", "\x1b[C": "right", "\x1bOC": "right",
        "\x1b[A": "up", "\x1bOA": "up", "\x1b[B": "down", "\x1bOB": "down",
        "\x1b[H": "home", "\x1bOH": "home", "\x1b[1~": "home", "\x01": "home",
        "\x1b[F": "end", "\x1bOF": "end", "\x1b[4~": "end", "\x05": "end",
        "\x1b[3~": "delete", "\x08": "backspace", "\x7f": "backspace",
        "\r": "enter", "\n": "enter", "\x1b": "esc",
        "\x03": "interrupt", "\x04": "eof", "\x1a": "eof"}
# What msvcrt.getwch gives after its \xe0 or \x00 prefix, as the sequence a
# POSIX terminal sends for the same key.
WINDOWS = {"K": "\x1b[D", "M": "\x1b[C", "H": "\x1b[A", "P": "\x1b[B",
           "G": "\x1b[H", "O": "\x1b[F", "S": "\x1b[3~"}
TOKEN = re.compile(r"\x1b(?:\[[0-9;]*[~A-Za-z]|O[A-Za-z])?|.", re.S)
# ponytail: a paste is told from typing by speed - its keys come closer together
# than this. Bracketed paste (\x1b[?2004h) would be exact, if a terminal ever
# splits a paste wider than this.
GRACE = 0.02
history = []                    # lines sent this session, for up and down
_numbers = itertools.count(1)


class Paste(object):
    def __init__(self, text):
        self.text = text
        self.label = "[Pasted text #%d +%d lines]" % (next(_numbers), text.count("\n") + 1)


class Line(object):
    """What is on the line: characters and Pastes, the cursor before units[at]."""

    def __init__(self, text=""):
        self.units = list(text)
        self.at = len(self.units)
        self.versions = history + [self.units]  # up and down walk these; edits stay until sent
        self.shown = len(history)

    def text(self):
        return "".join(getattr(u, "text", u) for u in self.units)

    def feed(self, burst):
        """Take what came in one go; returns the text once enter is pressed.
        A burst with a line break inside it is a paste and becomes one unit."""
        flat = burst.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if "\n" in flat:
            self.units.insert(self.at, Paste(flat))
            self.at += 1
            return None
        for key in TOKEN.findall(burst):
            name = KEYS.get(key)
            if name == "enter":
                if self.units:
                    history.append(self.units)
                return self.text()
            if name == "interrupt":
                raise KeyboardInterrupt
            if name == "eof" and not self.units:
                raise EOFError
            if name == "backspace" and self.at:
                self.at -= 1
                del self.units[self.at]
            elif name == "delete":
                del self.units[self.at:self.at + 1]
            elif name == "left":
                self.at = max(0, self.at - 1)
            elif name == "right":
                self.at = min(len(self.units), self.at + 1)
            elif name == "home":
                self.at = 0
            elif name == "end":
                self.at = len(self.units)
            elif name == "esc":
                self.units, self.at = [], 0
            elif name in ("up", "down"):
                self.versions[self.shown] = self.units
                self.shown = min(max(self.shown + (1 if name == "down" else -1), 0),
                                 len(self.versions) - 1)
                self.units = list(self.versions[self.shown])
                self.at = len(self.units)
            elif name is None and key >= " ":
                self.units.insert(self.at, key)
                self.at += 1
        return None


def read_line(style, prompt="› ", text=""):
    """input() that keeps a paste whole. `text` is on the line already (typed
    while the agent worked). Ctrl+c and ctrl+d raise as they do in input()."""
    if not sys.stdin.isatty():
        return text + input(style.cyan(prompt) + text)
    line, row, done = Line(text), 0, None
    with _reader() as (read, ready):
        try:
            while done is None:
                row = _draw(line, style, prompt, row)
                burst = read()
                while ready(GRACE):
                    burst += read()
                done = line.feed(burst)
        finally:
            line.at = len(line.units)          # what follows is printed below the line
            _draw(line, style, prompt, row)
    print()
    return done


def _draw(line, style, prompt, row):
    """Draw the line again from its first row, `row` rows above the cursor.
    Returns the row the cursor is left on."""
    cols = max(1, shutil.get_terminal_size().columns)
    shown = [getattr(u, "label", u) for u in line.units]
    end = _width(prompt + "".join(shown))
    cursor = _width(prompt + "".join(shown[:line.at]))
    out = ("\x1b[%dA" % row if row else "") + "\r\x1b[J" + style.cyan(prompt)
    out += "".join(style.grey(u.label) if isinstance(u, Paste) else u for u in line.units)
    if end % cols == 0:
        out += "\r\n"           # a full last row leaves the cursor hanging past its end
    up = end // cols - cursor // cols
    out += ("\x1b[%dA" % up if up else "") + "\r"
    out += "\x1b[%dC" % (cursor % cols) if cursor % cols else ""
    sys.stdout.write(out)
    sys.stdout.flush()
    return cursor // cols


def _width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


@contextlib.contextmanager
def _reader():
    """(read, ready): read() waits for the next key or chunk, ready(t) says
    whether more came within t seconds."""
    if os.name == "nt":
        import msvcrt

        def read():
            key = msvcrt.getwch()
            return WINDOWS.get(msvcrt.getwch(), "") if key in ("\x00", "\xe0") else key

        def ready(timeout):
            end = time.monotonic() + timeout
            while not msvcrt.kbhit():
                if time.monotonic() > end:
                    return False
                time.sleep(0.002)
            return True
        yield read, ready
        return
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    decode = codecs.getincrementaldecoder("utf-8")("ignore").decode   # a chunk can split a character
    try:
        yield (lambda: decode(os.read(fd, 4096)),
               lambda timeout: bool(select.select([fd], [], [], timeout)[0]))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
