# -*- coding: utf-8 -*-
"""Render the model's markdown in the terminal.

The model answers in markdown, so printing it raw leaves the reader to decode
`**bold**`, `[text](url)` and pipe tables by eye. Rendering happens line by line
so the answer still streams; only tables are held back, because their column
widths are not known until the block ends.
"""
import re
import shutil

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CODE = re.compile(r"`([^`]+)`")
LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)[^)]*\)")
BOLD_ITALIC = re.compile(r"\*\*\*(.+?)\*\*\*")
BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
ITALIC = re.compile(r"(?<![\*\w])\*([^\*\n]+)\*(?!\*)|(?<![_\w])_([^_\n]+)_(?![_\w])")
# `*` is excluded so a bold URL (**https://x/**) does not swallow its own closing
# markers - the stray `**` left behind then broke bold for the rest of the answer.
BARE_URL = re.compile(r"(?<![\(\[<])\bhttps?://[^\s<>\)\]\"'*]+")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
NUMBERED = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
QUOTE = re.compile(r"^\s*>\s?(.*)$")
RULE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
FENCE = re.compile(r"^\s*```(.*)$")
TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")

PLACEHOLDER = "\x00%d\x00"


def visible_len(text):
    return len(ANSI.sub("", text))


def inline(text, style):
    """Bold, italic, code and links. Code spans are pulled out first so their
    contents are never treated as markup."""
    spans = []

    def stash(rendered):
        spans.append(rendered)
        return PLACEHOLDER % (len(spans) - 1)

    text = CODE.sub(lambda m: stash(style.cyan(m.group(1))), text)
    text = LINK.sub(
        lambda m: stash(style.underline(style.cyan(m.group(1) or m.group(2)))
                        + style.dim(" (" + m.group(2) + ")")), text)
    text = BARE_URL.sub(lambda m: stash(style.underline(style.cyan(m.group(0)))), text)
    text = BOLD_ITALIC.sub(lambda m: style.bold(style.italic(m.group(1))), text)
    text = BOLD.sub(lambda m: style.bold(m.group(1) or m.group(2)), text)
    text = ITALIC.sub(lambda m: style.italic(m.group(1) or m.group(2)), text)
    for index, rendered in enumerate(spans):
        text = text.replace(PLACEHOLDER % index, rendered)
    return text


def table(rows, style, width=None):
    """Align a pipe table. Widths come from the plain text, styling is applied
    afterwards, so escape codes never count towards a column."""
    grid = [[cell.strip() for cell in _cells(row)] for row in rows]
    grid = [row for row in grid if row]
    if not grid:
        return []
    columns = max(len(row) for row in grid)
    grid = [row + [""] * (columns - len(row)) for row in grid]
    widths = [max(len(row[i]) for row in grid) for i in range(columns)]

    budget = (width or shutil.get_terminal_size((100, 25)).columns) - 4 - 3 * (columns - 1)
    if sum(widths) > budget > columns * 4:
        share = budget / float(sum(widths))
        widths = [max(4, int(w * share)) for w in widths]

    out = []
    for index, row in enumerate(grid):
        cells = []
        for column, cell in enumerate(row):
            plain = cell[:widths[column]]
            padding = " " * (widths[column] - len(plain))
            body = style.bold(plain) if index == 0 else inline(plain, style)
            cells.append(body + padding)
        out.append("  " + style.dim(" │ ").join(cells))
        if index == 0:
            out.append("  " + style.dim("─┼─".join("─" * w for w in widths)))
    return out


def _cells(row):
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return row.split("|")


class MarkdownStream(object):
    """Feed it chunks; it writes rendered lines as soon as they are complete."""

    def __init__(self, style, write):
        self.style = style
        self.write = write
        self.buffer = ""
        self.table_rows = []
        self.in_fence = False

    def feed(self, chunk):
        self.buffer += chunk
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line)

    def close(self):
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""
        self._flush_table()

    # ------------------------------------------------------------- internals
    def _flush_table(self):
        if self.table_rows:
            for line in table(self.table_rows, self.style):
                self.write(line + "\n")
            self.table_rows = []

    def _line(self, line):
        if self.in_fence:
            if FENCE.match(line):
                self.in_fence = False
            else:
                self.write("  " + self.style.dim("│ ") + self.style.cyan(line) + "\n")
            return

        fence = FENCE.match(line)
        if fence:
            self.in_fence = True
            self._flush_table()
            if fence.group(1).strip():
                self.write("  " + self.style.dim(fence.group(1).strip()) + "\n")
            return

        stripped = line.strip()
        if stripped.startswith("|") or (self.table_rows and "|" in stripped):
            if TABLE_SEP.match(stripped):
                return                       # the ---|--- row is only a marker
            self.table_rows.append(line)
            return
        self._flush_table()

        if not stripped:
            self.write("\n")
            return

        if RULE.match(line):
            width = min(shutil.get_terminal_size((100, 25)).columns - 2, 60)
            self.write(self.style.dim("─" * width) + "\n")
            return

        heading = HEADING.match(line)
        if heading:
            text = inline(heading.group(2), self.style)
            self.write("\n" + self.style.bold(self.style.cyan(text)) + "\n")
            return

        quote = QUOTE.match(line)
        if quote:
            self.write(self.style.dim("│ ") + self.style.dim(
                inline(quote.group(1), self.style)) + "\n")
            return

        bullet = BULLET.match(line)
        if bullet:
            self.write(bullet.group(1) + self.style.cyan("• ")
                       + inline(bullet.group(2), self.style) + "\n")
            return

        numbered = NUMBERED.match(line)
        if numbered:
            self.write(numbered.group(1) + self.style.cyan(numbered.group(2) + " ")
                       + inline(numbered.group(3), self.style) + "\n")
            return

        self.write(inline(line, self.style) + "\n")
