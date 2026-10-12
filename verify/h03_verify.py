# -*- coding: utf-8 -*-
"""h03: a backtracking regex engine without the re module."""
from common import Raises, check, forbids, load, report

forbids("rules", "myre.py", r"^\s*(import\s+re\b|from\s+re\b|import\s+.*\bre\b)")
m = load("myre")
ReError = getattr(m, "ReError", None)
check("rules", "ReError is a ValueError",
      lambda: isinstance(ReError, type) and issubclass(ReError, ValueError))

for pattern, text, value in [
        ("abc", "abc", True), ("a.c", "abc", True), ("a.c", "ac", False), ("abc", "abcd", False),
        ("a*b", "b", True), ("a*b", "aaab", True), ("a+b", "b", False), ("a+b", "aab", True),
        ("colou?r", "color", True), ("colou?r", "colour", True), ("colou?r", "colouur", False),
        ("[abc]+", "abcabc", True), ("[abc]+", "abd", False), ("[a-z]+[0-9]*", "abc123", True),
        ("[a-z]+[0-9]*", "abc", True), ("[^0-9]+", "abc", True), ("[^0-9]+", "a1", False),
        ("(ab)+c", "ababc", True), ("(ab)+c", "abac", False), ("(cat|dog)s?", "dogs", True),
        ("(cat|dog)s?", "cat", True), ("(cat|dog)s?", "cow", False), ("((a|b)c)+", "acbc", True),
        ("((a|b)c)+", "acc", False), ("cat|dog", "dog", True), ("a|b|c", "c", True),
        (r"\d+", "12345", True), (r"\d+", "12a", False), (r"\w+", "ab_9", True),
        (r"\s", " ", True), (r"a\.b", "a.b", True), (r"a\.b", "axb", False),
        (r"\*\[\\", "*[\\", True), ("x*", "", True), ("^abc$", "abc", True),
        ("a*a", "aaa", True), ("(a|ab)c", "abc", True), ("(a*)*b", "aab", True)]:
    check("match", "match(%r, %r)" % (pattern, text),
          lambda p=pattern, t=text: bool(m.match(p, t)), value)

for pattern, text, value in [
        ("^ab", "xab", None), ("ab$", "abx", None), ("a+", "caaab", (1, 4)), ("b", "abc", (1, 2)),
        ("x", "abc", None), ("[0-9]+", "ab123c", (2, 5)), ("cat|dog", "hotdog", (3, 6)),
        ("ab$", "xab", (1, 3)), ("^a", "abc", (0, 1)), (r"\d\d", "a1b22", (3, 5)),
        ("o+", "foo boo", (1, 3)), ("(ab)+", "xxababx", (2, 6)), ("a*", "bbb", (0, 0))]:
    check("search", "search(%r, %r)" % (pattern, text),
          lambda p=pattern, t=text: (lambda r: tuple(r) if r is not None else None)(m.search(p, t)),
          value)

for pattern in ["(", "[a", "*abc", "a**", "a)"]:
    check("errors", "%r raises ReError" % pattern, lambda p=pattern: m.match(p, "abc"),
          Raises(ReError or Exception))
report()
