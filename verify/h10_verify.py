# -*- coding: utf-8 -*-
"""h10: a dependency aware scheduler.

The task names no class, so the checker takes the module's first class that
has add() and topo(). run() may return the start events alone or together
with the makespan; the makespan is recomputed from the events either way.
Cycles are not built here: add() must refuse unknown dependencies, so the
API offers no way to make one."""
import inspect

from common import Raises, check, load, report

m = load("scheduler")
SchedError = getattr(m, "SchedError", None)
CLASSES = [c for _, c in inspect.getmembers(m, inspect.isclass)
           if hasattr(c, "add") and hasattr(c, "topo")]
check("rules", "a class with add() and topo()", lambda: bool(CLASSES))
Scheduler = CLASSES[0] if CLASSES else None

DIAMOND = [("a", (), 1, 0), ("b", ("a",), 2, 0), ("c", ("a",), 3, 0), ("d", ("b", "c"), 1, 0)]
MIXED = [("s", (), 1, 0), ("x", ("s",), 10, 0), ("y1", ("s",), 1, 0), ("y2", ("y1",), 1, 0),
         ("y3", ("y2",), 1, 0), ("end", ("x", "y3"), 1, 0)]
FAN = [("root", (), 1, 0)] + [("f%d" % i, ("root",), 1 + i % 3, i % 2) for i in range(8)] \
    + [("sink", tuple("f%d" % i for i in range(8)), 1, 0)]


def build(tasks):
    s = Scheduler()
    for name, deps, duration, priority in tasks:
        s.add(name, deps=deps, duration=duration, priority=priority)
    return s


def valid_topo(tasks):
    order = list(build(tasks).topo())
    return sorted(order) == sorted(t[0] for t in tasks) and all(
        order.index(dep) < order.index(name) for name, deps, _, _ in tasks for dep in deps)


for label, tasks in [("chain", [("t%d" % i, ("t%d" % (i - 1),) if i else (), 1, 0) for i in range(5)]),
                     ("diamond", DIAMOND), ("fan out", FAN), ("mixed", MIXED)]:
    check("topo", "%s: a valid order" % label, lambda t=tasks: valid_topo(t))
check("topo", "insertion order does not matter",
      lambda: list(build(DIAMOND).topo()) == list(build([DIAMOND[0], DIAMOND[2], DIAMOND[1],
                                                         DIAMOND[3]]).topo()))
check("topo", "higher priority first, then name",
      lambda: list(build([("c", (), 1, 0), ("a", (), 1, 0), ("b", (), 1, 5)]).topo()), ["b", "a", "c"])
check("cycle", "no cycle in a DAG", lambda: build(DIAMOND).cycle(), None)
check("waves", "diamond", lambda: [sorted(w) for w in build(DIAMOND).waves()], [["a"], ["b", "c"], ["d"]])
check("waves", "fan out", lambda: [len(w) for w in build(FAN).waves()], [1, 8, 1])
check("critical", "the longest duration, not the most tasks",
      lambda: (lambda r: (r[0], list(r[1])))(build(MIXED).critical_path()), (12, ["s", "x", "end"]))
check("critical", "diamond", lambda: build(DIAMOND).critical_path()[0], 5)


def schedule(tasks, workers):
    s = build(tasks)
    out = s.run(workers=workers)
    events = out[0] if isinstance(out, tuple) and len(out) == 2 and \
        isinstance(out[1], (int, float)) else out
    starts = dict((name, time) for time, name in events)
    durations = dict((t[0], t[2]) for t in tasks)
    return starts, durations, max(starts[n] + durations[n] for n in starts)


def respects(tasks, workers):
    starts, durations, _ = schedule(tasks, workers)
    if sorted(starts) != sorted(t[0] for t in tasks):
        return False
    for name, deps, _, _ in tasks:
        if any(starts[name] < starts[d] + durations[d] for d in deps):
            return False
    moments = sorted(set(starts.values()))
    return all(sum(1 for n in starts if starts[n] <= t < starts[n] + durations[n]) <= workers
               for t in moments)


for tasks, label in [(MIXED, "mixed"), (FAN, "fan out")]:
    for workers in (1, 2, 4):
        check("run", "%s, %d workers: dependencies and worker count respected" % (label, workers),
              lambda t=tasks, w=workers: respects(t, w))
check("run", "1 worker: makespan is the sum of durations",
      lambda: schedule(MIXED, 1)[2], sum(t[2] for t in MIXED))
check("run", "many workers: makespan is the critical path", lambda: schedule(MIXED, 8)[2], 12)
check("run", "fan out with 4 workers beats 1", lambda: schedule(FAN, 4)[2] < schedule(FAN, 1)[2])
check("errors", "a duplicate name raises SchedError",
      lambda: build([("a", (), 1, 0), ("a", (), 1, 0)]), Raises(SchedError or Exception))
check("errors", "an unknown dependency raises SchedError",
      lambda: build([("a", ("ghost",), 1, 0)]), Raises(SchedError or Exception))
report()
