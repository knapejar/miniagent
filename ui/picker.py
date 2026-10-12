# -*- coding: utf-8 -*-
"""A list to choose from with the arrow keys - what /resume shows."""
import contextlib
import os
import shutil
import sys

UP, DOWN, TOGGLE, ENTER, CANCEL = "up", "down", "toggle", "enter", "cancel"
KEYS = {"\x1b[A": UP, "\x1bOA": UP, "\x00H": UP, "\xe0H": UP, "k": UP,
        "\x1b[B": DOWN, "\x1bOB": DOWN, "\x00P": DOWN, "\xe0P": DOWN, "j": DOWN,
        "\x01": TOGGLE, "\r": ENTER, "\n": ENTER, "\x1b": CANCEL, "\x03": CANCEL,
        "q": CANCEL}
ROWS = 12


def choose(load, style, title):
    """`load(everything)` returns [(text, value)], ctrl+a flips `everything`.
    Returns the chosen value; None on esc or when stdin is not a terminal."""
    if not sys.stdin.isatty():
        return None
    everything, at, drawn = False, 0, 0
    rows = load(everything)
    try:
        with _keys() as read:
            while True:
                drawn = _draw(rows, at, drawn, style, "%s  %s  ·  ctrl+a %s  ·  ↑↓ enter esc" % (
                    title, "all projects" if everything else "this project",
                    "this project only" if everything else "all projects"))
                key = KEYS.get(read())
                if key == UP:
                    at = max(0, at - 1)
                elif key == DOWN:
                    at = min(len(rows) - 1, at + 1)
                elif key == TOGGLE:
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


def _draw(rows, at, drawn, style, header):
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
        lines.append("  " + style.dim("no saved sessions"))
    sys.stdout.write(out + "\n".join(lines) + "\n")
    sys.stdout.flush()
    return len(lines)


@contextlib.contextmanager
def _keys():
    """A reader of single keypresses; an arrow comes back as one string."""
    if os.name == "nt":
        import msvcrt

        def read():
            key = msvcrt.getwch()
            return key + msvcrt.getwch() if key in ("\x00", "\xe0") else key
        yield read
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield lambda: os.read(fd, 16).decode("utf-8", "ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
