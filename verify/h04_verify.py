# -*- coding: utf-8 -*-
"""h04: a spreadsheet with formulas, ranges, dependencies and cycle detection."""
from common import Raises, check, forbids, load, report

forbids("rules", "sheet.py", calls=("eval", "exec", "compile"))
m = load("sheet")
CycleError = getattr(m, "CycleError", None)
CellError = getattr(m, "CellError", None)


def sheet(**cells):
    s = m.Sheet()
    for ref, value in cells.items():
        s.set(ref, value)
    return s


check("literals", "a number", lambda: sheet(A1=5).get("A1"), 5)
check("literals", "a float", lambda: sheet(A1=2.5).get("A1"), 2.5)
check("literals", "a string", lambda: sheet(A1="hello").get("A1"), "hello")
check("literals", "an empty cell is 0", lambda: sheet().get("Z9"), 0)
check("formulas", "=A1+B1", lambda: sheet(A1=2, B1=3, C1="=A1+B1").get("C1"), 5)
check("formulas", "=A1*2+3", lambda: sheet(A1=2, C1="=A1*2+3").get("C1"), 7)
check("formulas", "=(A1+B1)/2", lambda: sheet(A1=2, B1=3, C1="=(A1+B1)/2").get("C1"), 2.5)
check("formulas", "=A1-B1*2", lambda: sheet(A1=10, B1=3, C1="=A1-B1*2").get("C1"), 4)
check("formulas", "an empty cell counts as 0", lambda: sheet(A1="=Z99+1").get("A1"), 1)
check("formulas", "a constant formula", lambda: sheet(A1="=1+2*3").get("A1"), 7)


def chain():
    s = sheet(A1=1, B1="=A1+1", C1="=B1*10")
    first = s.get("C1")
    s.set("A1", 4)
    return first, s.get("C1")


check("chains", "a change reaches the end of a chain", chain, (20, 50))


def late_dependency():
    s = sheet(B1="=A1*2")
    s.get("B1")
    s.set("A1", 21)
    return s.get("B1")


check("chains", "a cell set after the formula", late_dependency, 42)

col = dict(("A%d" % i, i) for i in range(1, 6))
for formula, value in [("=SUM(A1:A5)", 15), ("=AVG(A1:A4)", 2.5), ("=MIN(A1:A5)", 1),
                       ("=MAX(A1:A5)", 5), ("=COUNT(A1:A5)", 5), ("=SUM(A2:A3)+1", 6)]:
    check("ranges", formula, lambda f=formula: sheet(Z1=f, **col).get("Z1"), value)
grid = dict(("%s%d" % (c, r), (r - 1) * 3 + i + 1)
            for r in range(1, 4) for i, c in enumerate("ABC"))
check("ranges", "a 2D range =SUM(A1:C3)", lambda: sheet(Z1="=SUM(A1:C3)", **grid).get("Z1"), 45)
check("ranges", "=MAX(A1:C3)", lambda: sheet(Z1="=MAX(A1:C3)", **grid).get("Z1"), 9)
check("mixed", "=SUM(A1:A3)*2 + MAX(B1:B2)",
      lambda: sheet(A1=1, A2=2, A3=3, B1=4, B2=9, Z1="=SUM(A1:A3)*2 + MAX(B1:B2)").get("Z1"), 21)


def range_update():
    s = sheet(Z1="=SUM(A1:A5)", **col)
    s.get("Z1")
    s.set("A3", 100)
    return s.get("Z1")


check("mixed", "a range sees a changed cell", range_update, 112)
check("columns", "AA1", lambda: sheet(AA1=7, B1="=AA1*2").get("B1"), 14)
check("columns", "AB12", lambda: sheet(AB12=3, A1="=AB12+AA1").get("A1"), 3)


def cycle(*cells):
    s = m.Sheet()
    for ref, value in cells:
        s.set(ref, value)
    return s.get(cells[0][0])


check("cycles", "A1 <-> B1 raises CycleError", lambda: cycle(("A1", "=B1"), ("B1", "=A1")),
      Raises(CycleError or Exception))
check("cycles", "a 3-cell cycle raises CycleError",
      lambda: cycle(("A1", "=B1+1"), ("B1", "=C1"), ("C1", "=A1")), Raises(CycleError or Exception))
check("cycles", "a self reference raises CycleError", lambda: cycle(("A1", "=A1+1"),),
      Raises(CycleError or Exception))


def names_a_cell():
    try:
        cycle(("A1", "=B1"), ("B1", "=A1"))
    except Exception as e:                                # noqa: BLE001
        return "A1" in str(e) or "B1" in str(e)
    return False


check("cycles", "the CycleError names a cell", names_a_cell)
check("errors", "dividing by zero raises CellError",
      lambda: sheet(A1=0, B1="=10/A1").get("B1"), Raises(CellError or Exception))
check("errors", "CycleError and CellError are different",
      lambda: CycleError is not None and CellError is not None and CycleError is not CellError)
report()
