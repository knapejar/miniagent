# -*- coding: utf-8 -*-
"""A list to choose from with the arrow keys - what /resume and /model show."""
import contextlib
import os
import shutil
import sys
import time

UP, DOWN, TOGGLE, ENTER, CANCEL = "up", "down", "toggle", "enter", "cancel"
KEYS = {"\x1b[A": UP, "\x1bOA": UP, "\x00H": UP, "\xe0H": UP, "k": UP,
        "\x1b[B": DOWN, "\x1bOB": DOWN, "\x00P": DOWN, "\xe0P": DOWN, "j": DOWN,
        "\x01": TOGGLE, "\r": ENTER, "\n": ENTER, "\x1b": CANCEL, "\x03": CANCEL,
        "q": CANCEL}
ROWS = 12


def choose(load, style, title, toggle=None, empty="nothing to choose", at=0, busy=None):
    """`load(everything)` returns [(text, value)]; with toggle=(off label, on
    label) ctrl+a flips `everything`. The cursor starts on row `at`. While
    `busy()` is true the rows are still filling in: they are loaded again
    and redrawn every 0.1 s.
    Returns the chosen value; None on esc or when stdin is not a terminal."""
    if not sys.stdin.isatty():
        return None
    everything, drawn = False, 0
    rows = load(everything)
    try:
        with _keys() as read:
            while True:
                waiting = busy is not None and busy()   # before load, or the last answer is missed
                if busy is not None:
                    rows = load(everything)
                header = title
                if toggle:
                    header += "  %s  ·  ctrl+a %s" % (toggle[everything], toggle[not everything])
                drawn = _draw(rows, at, drawn, style, header + "  ·  ↑↓ enter esc", empty)
                key = KEYS.get(read(0.1) if waiting else read())
                if key == UP:
                    at = max(0, at - 1)
                elif key == DOWN:
                    at = min(len(rows) - 1, at + 1)
                elif key == TOGGLE and toggle:
                    everything, at = not everything, 0
                    rows = load(everything)
                elif key == ENTER and rows:
                    return rows[at][1]
                elif key == CANCEL:
                    return None
    except KeyboardInterrupt:
        return None
    finally:
        _draw([], 0, drawn, style, None)        # take the list off the screen


def _draw(rows, at, drawn, style, header, empty=""):
    """Redraw in place over the `drawn` lines printed last time; returns the new count."""
    out = ("\x1b[%dA" % drawn if drawn else "") + "\r\x1b[J"
    if header is None:
        sys.stdout.write(out)
        sys.stdout.flush()
        return 0
    width = shutil.get_terminal_size().columns - 4     # no wrap, or the count is off
    top = max(0, min(at - ROWS // 2, len(rows) - ROWS))
    lines = ["  " + style.dim(header[:width])]
    for i, (text, _value) in enumerate(rows[top:top + ROWS], top):
        text = text[:width]
        lines.append(style.cyan("› " + text) if i == at else "  " + text)
    if not rows:
        lines.append("  " + style.dim(empty))
    sys.stdout.write(out + "\n".join(lines) + "\n")
    sys.stdout.flush()
    return len(lines)


@contextlib.contextmanager
def _keys():
    """A reader of single keypresses; an arrow comes back as one string.
    read(timeout) gives None when no key came within timeout seconds."""
    if os.name == "nt":
        import msvcrt

        def read(timeout=None):
            if timeout is not None:
                end = time.monotonic() + timeout
                while not msvcrt.kbhit():
                    if time.monotonic() >= end:
                        return None
                    time.sleep(0.01)
            key = msvcrt.getwch()
            return key + msvcrt.getwch() if key in ("\x00", "\xe0") else key
        yield read
        return
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def read(timeout=None):
        if timeout is not None and not select.select([fd], [], [], timeout)[0]:
            return None
        return os.read(fd, 16).decode("utf-8", "ignore")
    try:
        yield read
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
