# -*- coding: utf-8 -*-
"""h09: a date library without datetime. The checker itself compares against
datetime; ordinals are compared as differences, so 0001-01-01 may be day 0 or 1."""
import datetime

from common import Raises, check, forbids, load, report

forbids("rules", "mydate.py", r"^\s*(import|from)\s+(datetime|time|calendar)\b")
m = load("mydate")
Date = m.Date
DateError = getattr(m, "DateError", None)
check("rules", "DateError is a ValueError",
      lambda: isinstance(DateError, type) and issubclass(DateError, ValueError))


def either(name):
    return getattr(m, name, None) or getattr(Date, name)


def same(d, y, mo, day):
    return str(d) == "%04d-%02d-%02d" % (y, mo, day)


for year, leap in [(1900, False), (2000, True), (2023, False), (2024, True), (2100, False)]:
    check("leap", "is_leap(%d)" % year, lambda y=year: bool(either("is_leap")(y)), leap)
for args in [(2023, 2, 29), (2023, 13, 1), (2023, 4, 31), (2023, 0, 10), (2023, 1, 0)]:
    check("validation", "Date%r raises" % (args,), lambda a=args: Date(*a), Raises(DateError or Exception))
check("validation", "2024-02-29 is fine", lambda: str(Date(2024, 2, 29)), "2024-02-29")
check("str", "zero padded", lambda: str(Date(2024, 3, 9)), "2024-03-09")

for args in [(1, 1, 2), (1600, 2, 29), (1700, 3, 1), (1899, 12, 31), (1970, 1, 1), (2000, 2, 29),
             (2024, 12, 31), (2100, 3, 1)]:
    check("ordinal", "%r days from 0001-01-01" % (args,),
          lambda a=args: Date(*a).to_ordinal() - Date(1, 1, 1).to_ordinal(),
          datetime.date(*args).toordinal() - 1)
    check("ordinal", "%r round trips" % (args,),
          lambda a=args: str(either("from_ordinal")(Date(*a).to_ordinal())), str(datetime.date(*args)))

for args, n in [((2024, 1, 31), 1), ((2023, 12, 31), 1), ((2024, 2, 28), 1), ((2023, 2, 28), 1),
                ((2024, 3, 1), -1), ((2024, 5, 10), -400), ((2000, 1, 1), 10000), ((2024, 1, 1), 0)]:
    check("add_days", "%r + %d" % (args, n), lambda a=args, n=n: str(Date(*a).add_days(n)),
          str(datetime.date(*args) + datetime.timedelta(days=n)))
for args, n, want in [((2024, 1, 31), 1, "2024-02-29"), ((2023, 1, 31), 1, "2023-02-28"),
                      ((2024, 3, 31), -1, "2024-02-29"), ((2023, 12, 15), 13, "2025-01-15"),
                      ((2024, 5, 10), 24, "2026-05-10"), ((2024, 1, 15), -1, "2023-12-15")]:
    check("add_months", "%r + %d months" % (args, n),
          lambda a=args, n=n: str(Date(*a).add_months(n)), want)
for args in [(2000, 1, 1), (1970, 1, 1), (2024, 3, 9), (1900, 1, 1), (2023, 12, 25), (2100, 6, 1)]:
    check("weekday", "%r" % (args,), lambda a=args: Date(*a).weekday(), datetime.date(*args).weekday())
check("diff", "diff_days is 366 across 2024",
      lambda: abs(Date(2025, 1, 1).diff_days(Date(2024, 1, 1))), 366)
check("diff", "diff_days the other way",
      lambda: Date(2024, 1, 1).diff_days(Date(2025, 1, 1)) == -Date(2025, 1, 1).diff_days(Date(2024, 1, 1)))
check("compare", "==", lambda: Date(2024, 3, 9) == Date(2024, 3, 9))
check("compare", "<", lambda: Date(2024, 3, 8) < Date(2024, 3, 9))
check("compare", ">", lambda: Date(2025, 1, 1) > Date(2024, 12, 31))
check("parse", "YYYY-MM-DD", lambda: str(either("parse")("2024-03-09")), "2024-03-09")
check("parse", "DD.MM.YYYY", lambda: str(either("parse")("09.03.2024")), "2024-03-09")
for text in ["2023-13-01", "31/12/2023", "abc", "2023-02-29"]:
    check("parse", "%r raises DateError" % text, lambda t=text: either("parse")(t),
          Raises(DateError or Exception))
for fmt, want in [("%Y-%m-%d", "2024-03-09"), ("%B", "March"), ("%A", "Saturday"),
                  ("%d.%m.%Y", "09.03.2024")]:
    check("format", fmt, lambda f=fmt: Date(2024, 3, 9).format(f), want)
report()
