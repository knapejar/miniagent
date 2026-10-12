# -*- coding: utf-8 -*-
"""Run benchmark tasks and score them with the independent checkers.

    python verify/bench.py tasks/hard/h04_spreadsheet.txt tasks/edit/*.txt
    python verify/bench.py --rev 3f042c4 --label before tasks/edit/*.txt

Each task runs in its own empty workspace under ~/.miniagent/bench/<label>/,
with --no-memory, so no run sees another one's notes. A task file may start with
`fixture: git <rev>`: the workspace then starts as that revision of this
repository - the code the task edits. --rev runs the harness of an older commit,
so before/after are compared on the same tasks while the working tree moves on.

The score is the checker's (verify/<id>_verify.py <workspace>, last line
`SCORE: passed/total`), never the agent's own claim. Results are appended to
~/.miniagent/bench/results.jsonl, one line per run.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import paths                                     # noqa: E402

VERIFY = os.path.join(ROOT, "verify")
SCORE = re.compile(r"(?:SCORE|CELKEM):\s*(\d+)/(\d+)")


def archive(rev, target):
    """`git archive rev` unpacked into target (skipped when already there)."""
    if os.path.isdir(target) and os.listdir(target):
        return
    os.makedirs(target, exist_ok=True)
    blob = subprocess.run(["git", "-C", ROOT, "archive", rev], capture_output=True,
                          check=True).stdout
    subprocess.run(["tar", "-x", "-C", target], input=blob, check=True)


def read_task(path):
    """(task text, fixture rev or None)."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    match = re.match(r"fixture:\s*git\s+(\S+)", lines[0]) if lines else None
    if match:
        return "\n".join(lines[1:]).strip(), match.group(1)
    return "\n".join(lines), None


def summarize(trace):
    """Steps, how the run ended and what it cost, from its JSONL trace."""
    out = {"steps": 0, "end": "none", "edits": 0, "edit_errors": 0, "crops": 0,
           "tokens": 0}
    try:
        with open(trace, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    except (OSError, ValueError):
        return out
    seen = set()
    for e in events:
        ev = e.get("ev")
        seen.add(ev)
        if ev == "step":
            out["steps"] = e["step"]
            if e.get("tool") in ("edit", "write"):
                out["edits"] += 1
                out["edit_errors"] += str(e.get("obs", "")).startswith("error")
        elif ev in ("crop", "crop_on_overflow"):
            out["crops"] += 1
        if e.get("metrics"):
            out["tokens"] = e["metrics"].get("total_tokens", out["tokens"])
    # a forced answer also logs "final", so the worse ending wins
    for ev, end in (("limit", "limit"), ("force_answer", "forced"), ("error", "error"),
                    ("asked", "asked"), ("final", "final")):
        if ev in seen:
            out["end"] = end
            break
    return out


def check(task_id, workspace):
    checker = os.path.join(VERIFY, "%s_verify.py" % task_id)
    if not os.path.exists(checker):
        return None, None, "no checker"
    try:
        done = subprocess.run([sys.executable, checker, workspace], cwd=workspace,
                              capture_output=True, timeout=300,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        text = done.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 0, 1, "checker timed out"
    found = SCORE.findall(text)
    if not found:
        return 0, 1, text.strip()[-300:]
    passed, total = map(int, found[-1])
    return passed, total, text.strip()[-600:]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tasks", nargs="+", help="task files (globs work too)")
    p.add_argument("--label", default=time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--rev", help="run the harness of this git revision")
    p.add_argument("--profile", default="local")
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--timeout", type=int, default=2700, help="seconds per run")
    p.add_argument("--extra", default="", help="more miniagent flags, one string")
    p.add_argument("--rescore", action="store_true",
                   help="only run the checkers again on the label's workspaces")
    a = p.parse_args()

    base = os.path.join(paths.home(), "bench", a.label)
    if a.rescore:
        for workspace in sorted(glob.glob(os.path.join(base, "*", "ws"))):
            name = os.path.basename(os.path.dirname(workspace))
            passed, total, _ = check(name.split("_")[0], workspace)
            print("%-28s %s" % (name, "-" if passed is None else "%d/%d" % (passed, total)))
        return
    harness = ROOT
    if a.rev:
        harness = os.path.join(base, "harness")
        archive(a.rev, harness)
    files = [f for pattern in a.tasks for f in sorted(glob.glob(pattern))]
    rows = []
    for task_file in files:
        stem = os.path.splitext(os.path.basename(task_file))[0]
        task_id = stem.split("_")[0]
        text, fixture = read_task(task_file)
        for n in range(1, a.repeat + 1):
            name = stem if a.repeat == 1 else "%s_r%d" % (stem, n)
            run_dir = os.path.join(base, name)
            workspace = os.path.join(run_dir, "ws")
            if os.path.exists(workspace):
                print("skip %s - %s exists" % (name, workspace))
                continue
            os.makedirs(workspace)
            if fixture:
                archive(fixture, workspace)
            with open(os.path.join(run_dir, "task.txt"), "w", encoding="utf-8") as fh:
                fh.write(text)
            run_name = "bench_%s_%s" % (a.label, name)
            argv = [sys.executable, os.path.join(harness, "miniagent.py"), "-C", workspace,
                    "--profile", a.profile, "--no-memory", "--no-color",
                    "--max-steps", str(a.max_steps), "--run-name", run_name,
                    "-f", os.path.join(run_dir, "task.txt")] + a.extra.split()
            print("run  %-28s" % name, end="", flush=True)
            started = time.time()
            with open(os.path.join(run_dir, "stdout.txt"), "wb") as sink:
                try:
                    subprocess.run(argv, stdout=sink, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, timeout=a.timeout,
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                    timed_out = False
                except subprocess.TimeoutExpired:
                    timed_out = True
            seconds = time.time() - started
            row = summarize(os.path.join(paths.runs_dir(workspace), run_name + ".jsonl"))
            if timed_out:
                row["end"] = "timeout"
            passed, total, detail = check(task_id, workspace)
            row.update(label=a.label, rev=a.rev or "worktree", task=name, extra=a.extra,
                       passed=passed, total=total, seconds=round(seconds),
                       detail=detail, at=time.strftime("%Y-%m-%d %H:%M"))
            rows.append(row)
            with open(os.path.join(paths.home(), "bench", "results.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(" %s" % line(row), flush=True)
    if rows:
        print("\n%-28s %6s %6s %6s %8s %6s" % ("task", "score", "steps", "end", "edits", "min"))
        for row in rows:
            print("%-28s %s" % (row["task"], line(row)))


def line(row):
    score = "-" if row["passed"] is None else "%d/%d" % (row["passed"], row["total"])
    return "%6s %6d %6s %4d/%-3d %6.1f" % (score, row["steps"], row["end"],
                                           row["edits"] - row["edit_errors"], row["edits"],
                                           row["seconds"] / 60.0)


if __name__ == "__main__":
    main()
