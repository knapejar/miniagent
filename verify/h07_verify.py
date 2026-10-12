# -*- coding: utf-8 -*-
"""h07: an event-sourced ledger with invariants, history, replay and snapshots."""
from common import Raises, check, load, report

m = load("ledger")
LedgerError = getattr(m, "LedgerError", None)

EVENTS = [{"type": "open", "acc": "A", "t": 1},
          {"type": "deposit", "acc": "A", "amount": 100, "t": 2},
          {"type": "withdraw", "acc": "A", "amount": 30, "t": 3},
          {"type": "open", "acc": "B", "t": 4},
          {"type": "transfer", "frm": "A", "to": "B", "amount": 50, "t": 5},
          {"type": "deposit", "acc": "B", "amount": 5, "t": 6}]


def ledger(events=EVENTS):
    led = m.Ledger()
    for event in events:
        led.apply(dict(event))
    return led


def restored(snap):
    if hasattr(m, "restore"):
        return m.restore(snap)
    fresh = m.Ledger()
    out = fresh.restore(snap)
    return out if isinstance(out, m.Ledger) else fresh


check("balances", "A after everything", lambda: ledger().balance("A"), 20)
check("balances", "B after everything", lambda: ledger().balance("B"), 55)
check("balances", "total", lambda: ledger().total(), 75)
check("balances", "accounts", lambda: sorted(ledger().accounts()), ["A", "B"])
check("balances", "a fresh account is 0", lambda: ledger(EVENTS[:4]).balance("B"), 0)
for at, value in [(1, 0), (2, 100), (3, 70), (4, 70), (5, 20), (99, 20)]:
    check("history", "balance(A, at=%d)" % at, lambda at=at: ledger().balance("A", at=at), value)
check("history", "balance(B, at=5)", lambda: ledger().balance("B", at=5), 50)
check("history", "history(B) holds the transfer and not the withdrawal",
      lambda: [e["type"] for e in ledger().history("B")][-2:] == ["transfer", "deposit"]
      and all(e["type"] != "withdraw" for e in ledger().history("B")))
check("history", "history(A) is in order",
      lambda: [e["t"] for e in ledger().history("A")] == sorted(e["t"] for e in ledger().history("A")))

for label, events in [
        ("unknown account", [{"type": "deposit", "acc": "Z", "amount": 1, "t": 1}]),
        ("opening twice", [EVENTS[0], {"type": "open", "acc": "A", "t": 2}]),
        ("zero amount", [EVENTS[0], {"type": "deposit", "acc": "A", "amount": 0, "t": 2}]),
        ("negative amount", [EVENTS[0], {"type": "deposit", "acc": "A", "amount": -5, "t": 2}]),
        ("overdrawn withdrawal", [EVENTS[0], {"type": "withdraw", "acc": "A", "amount": 1, "t": 2}]),
        ("overdrawn transfer", EVENTS[:4] + [dict(EVENTS[4], amount=500)]),
        ("time not increasing", [EVENTS[0], dict(EVENTS[1], t=1)]),
        ("transfer to unknown", EVENTS[:2] + [{"type": "transfer", "frm": "A", "to": "Q",
                                               "amount": 1, "t": 3}])]:
    check("errors", label, lambda events=events: ledger(events), Raises(LedgerError or Exception))


def rejected_event_changes_nothing():
    led = ledger()
    try:
        led.apply({"type": "withdraw", "acc": "A", "amount": 999, "t": 50})
    except Exception:                                     # noqa: BLE001
        pass
    return led.balance("A"), led.total()


check("errors", "a rejected event changes nothing", rejected_event_changes_nothing, (20, 75))
check("replay", "replay equals the incremental ledger",
      lambda: (m.replay([dict(e) for e in EVENTS]).balance("A"),
               m.replay([dict(e) for e in EVENTS]).total()), (20, 75))


def snapshot_round_trip():
    led = ledger()
    copy = restored(led.snapshot())
    return [copy.balance("A"), copy.balance("B"), copy.total(), copy.balance("A", at=3),
            sorted(copy.accounts())]


check("snapshot", "restored answers identically", snapshot_round_trip,
      [20, 55, 75, 70, ["A", "B"]])


def snapshot_is_independent():
    led = ledger()
    copy = restored(led.snapshot())
    led.apply({"type": "deposit", "acc": "A", "amount": 1000, "t": 10})
    copy.apply({"type": "deposit", "acc": "B", "amount": 1, "t": 10})
    return led.balance("A"), copy.balance("A"), copy.balance("B")


check("snapshot", "both go on independently", snapshot_is_independent, (1020, 20, 56))
report()
