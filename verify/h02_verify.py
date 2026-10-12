# -*- coding: utf-8 -*-
"""h02: expression interpreter without eval."""
from common import Raises, check, forbids, load, report

forbids("rules", "calc.py", r"(?<![\w.])(eval|exec|compile)\s*\(")
m = load("calc")
CalcError = getattr(m, "CalcError", None)
check("rules", "CalcError is an Exception", lambda: isinstance(CalcError, type))
check("tokenize", "tokenize('12+ab*(3)') has 7 tokens", lambda: len(m.tokenize("12+ab*(3)")), 7)
check("tokenize", "tokenize('min(1, 2)') has 6 tokens", lambda: len(m.tokenize("min(1, 2)")), 6)

for src, value in [("2+3*4", 14), ("(2+3)*4", 20), ("2^3^2", 512), ("-2^2", -4), ("10-2-3", 5),
                   ("2*3^2", 18), ("1+2*3-4/2", 5), ("-5+3", -2), ("--5", 5), ("3*-2", -6),
                   ("7%3", 1), ("7/2", 3.5), ("6/3", 2), ("2^0.5", 2 ** 0.5), ("1.5*4", 6.0),
                   ("((1))", 1), ("2^-1", 0.5)]:
    check("arith", src, lambda s=src: m.evaluate(s), value)

check("vars", "x = 5; x*2", lambda: m.evaluate("x = 5; x*2"), 10)
check("vars", "env passed in", lambda: m.evaluate("y*2", {"y": 4}), 8)
check("vars", "assignment returns the value", lambda: m.evaluate("z = 3"), 3)


def stored():
    env = {}
    m.evaluate("a = 7", env)
    return env.get("a")


check("vars", "assignment stores into env", stored, 7)
check("vars", "a; b; c returns the last", lambda: m.evaluate("a = 1; b = a + 1; b * 10"), 20)

for src, value in [("min(3,1)", 1), ("max(2,9)", 9), ("abs(-4)", 4), ("sqrt(16)", 4),
                   ("pow(2,10)", 1024), ("min(2+3, max(1, 2)*4)", 5), ("max(abs(-3), 2)", 3)]:
    check("functions", src, lambda s=src: m.evaluate(s), value)

for src in ["1/0", "5%0", "q+1", "foo(1)", "sqrt(1,2)", "(1+2", "1+2)", "1 2", "2*"]:
    check("errors", "%s raises CalcError" % src, lambda s=src: m.evaluate(s),
          Raises(CalcError or Exception))
report()
