# -*- coding: utf-8 -*-
"""h06: lcs / diff / unified diff / patch without difflib.

apply_patch may take and return the original as a list of lines or as text;
both are accepted, the round trip is what is checked."""
import random
import re

from common import Raises, check, forbids, load, report

forbids("rules", "mydiff.py", r"^\s*(import\s+difflib\b|from\s+difflib\b)")
m = load("mydiff")
PatchError = getattr(m, "PatchError", None)


def lines(value):
    return value.splitlines() if isinstance(value, str) else list(value)


def subsequence(sub, seq):
    it = iter(seq)
    return all(any(x == y for y in it) for x in sub)


def is_lcs(a, b, length):
    got = list(m.lcs(a, b))
    return len(got) == length and subsequence(got, a) and subsequence(got, b)


check("lcs", "identical lists", lambda: list(m.lcs(list("abc"), list("abc"))), list("abc"))
check("lcs", "disjoint lists", lambda: list(m.lcs(["a"], ["b"])), [])
check("lcs", "one empty", lambda: list(m.lcs([], ["a"])), [])
check("lcs", "ABCBDAB / BDCABA has length 4", lambda: is_lcs(list("ABCBDAB"), list("BDCABA"), 4))
check("lcs", "lines", lambda: is_lcs(["x", "a", "b", "y", "c"], ["a", "q", "b", "c"], 3))

check("diff", "no change gives an empty diff", lambda: list(m.diff(list("abc"), list("abc"))), [])
check("diff", "a change gives a hunk", lambda: len(m.diff(list("abc"), list("aXc"))) >= 1)


def header(a, b):
    return re.findall(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", m.to_unified(a, b))


L5 = ["l1", "l2", "l3", "l4", "l5"]
check("unified", "a change in the middle: @@ -1,3 +1,3 @@",
      lambda: header(["a", "b", "c"], ["a", "X", "c"]), [("1", "3", "1", "3")])
check("unified", "an append: @@ -3,3 +3,4 @@", lambda: header(L5, L5 + ["new"]), [("3", "3", "3", "4")])
check("unified", "the first line deleted: @@ -1,4 +1,3 @@",
      lambda: header(["x"] + L5[:4], L5[:4]), [("1", "4", "1", "3")])
L20 = ["line%d" % i for i in range(20)]


def changed(*at):
    return [("CHANGED%d" % i if i in at else x) for i, x in enumerate(L20)]


check("unified", "far apart changes are two hunks", lambda: len(header(L20, changed(1, 18))), 2)
check("unified", "close changes are one hunk", lambda: len(header(L20, changed(5, 8))), 1)
check("unified", "lines carry their prefixes",
      lambda: all(x in m.to_unified(["a", "b", "c"], ["a", "X", "c"]).splitlines()
                  for x in (" a", "-b", "+X", " c")))

rng = random.Random(4242)
for n in range(20):
    a = [rng.choice("abcde") for _ in range(30)]
    b = list(a)
    for _ in range(rng.randint(1, 6)):
        op, i = rng.choice("ird"), rng.randrange(len(b))
        if op == "i":
            b.insert(i, rng.choice("xyz"))
        elif op == "r":
            b[i] = rng.choice("xyz")
        elif len(b) > 1:
            del b[i]
    check("round trip", "pair %d" % n,
          lambda a=a, b=b: lines(m.apply_patch(a, m.to_unified(a, b))) == b)
check("round trip", "as text", lambda: lines(m.apply_patch("a\nb\nc\n", m.to_unified(
    ["a", "b", "c"], ["a", "B", "c"]))) == ["a", "B", "c"])
check("errors", "a patch for another text raises PatchError",
      lambda: m.apply_patch(["q", "r", "s"], m.to_unified(["a", "b", "c"], ["a", "X", "c"])),
      Raises(PatchError or Exception))
report()
