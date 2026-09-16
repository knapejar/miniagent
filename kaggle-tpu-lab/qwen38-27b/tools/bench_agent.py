"""Agent-loop benchmark: what an agent actually waits for, and whether cached state stays correct.

    python tools/bench_agent.py --label baseline --out runs/baseline.json
    python tools/bench_agent.py --agents 8 --turns 16 --tool-tokens 4000 --label prefix-cache

Each simulated agent resends its whole, growing conversation every turn (system prompt, then a
tool result of --tool-tokens per turn), exactly like Claude Code or miniagent. Per turn it records
TTFT (dominated by prefill of the history: what prefix caching should cut), prompt tokens and
decode speed. Every tool result hides a random value; later turns ask for an older one, so a
server that restores the wrong cached state answers wrong (the "accuracy garbles" failure mode).

Default base URL is the local proxy (http://127.0.0.1:8080/v1); for a raw tunnel set KTL_API_KEY.
Standard library only."""
import argparse
import json
import os
import random
import statistics
import threading
import time
import urllib.request

WORDS = ("def return self value index buffer request tensor layer cache block state token "
         "scheduler prefill decode config import class while for if else raise yield async "
         "await lambda dict list tuple none true false parse render stream chunk offset").split()


def filler(rng, tokens):
    """Code-like text of roughly `tokens` tokens (~0.75 words per token for this vocabulary)."""
    lines, words = [], int(tokens * 0.75)
    while words > 0:
        n = min(words, rng.randint(6, 14))
        lines.append("    " * rng.randint(0, 3) + " ".join(rng.choice(WORDS) for _ in range(n)))
        words -= n
    return "\n".join(lines)


def chat(base, model, messages, max_tokens, timeout):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.6, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("KTL_API_KEY", "x")})
    t0, first, text, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece and first is None:
                    first = time.time()
                text.append(piece)
    end = time.time()
    first = first or end
    completion = usage.get("completion_tokens", 0)
    return {"ttft": first - t0, "total": end - t0, "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": completion,
            "decode_tps": (completion - 1) / (end - first) if completion > 1 and end > first else 0.0,
            "text": "".join(text)}


def agent(i, args, results):
    rng = random.Random(args.seed + i)
    secrets_by_turn = {}
    messages = [{"role": "system", "content": "You are a coding agent. Answer tersely.\n\n"
                 + filler(rng, args.system_tokens)}]
    for turn in range(1, args.turns + 1):
        secret = "%06d" % rng.randint(0, 999999)
        secrets_by_turn[turn] = secret
        ask = turn - 2 if turn > 2 else None
        question = ("Reply with exactly the value of KEY-%d, digits only." % ask) if ask else "Reply with OK."
        content = ("Tool result (turn %d):\n%s\nKEY-%d = %s\n%s\n\n%s"
                   % (turn, filler(rng, args.tool_tokens // 2), turn, secret,
                      filler(rng, args.tool_tokens // 2), question))
        messages.append({"role": "user", "content": content})
        try:
            r = chat(args.base, args.model, messages, args.max_tokens, args.timeout)
        except Exception as e:                       # a failed turn is a result too
            results.append({"agent": i, "turn": turn, "error": "%s: %s" % (type(e).__name__, e)})
            messages.append({"role": "assistant", "content": "error"})
            continue
        answer = r.pop("text")
        r.update(agent=i, turn=turn)
        if ask:
            r["correct"] = secrets_by_turn[ask] in answer
        results.append(r)
        messages.append({"role": "assistant", "content": answer})


def summarize(results, wall, args):
    ok = [r for r in results if "error" not in r]
    later = [r for r in ok if r["turn"] > 1]           # turn 1 has nothing to reuse
    checks = [r for r in ok if "correct" in r]
    by_turn = {}
    for r in ok:
        by_turn.setdefault(r["turn"], []).append(r)
    med = lambda xs: round(statistics.median(xs), 3) if xs else None
    return {
        "label": args.label, "base": args.base, "agents": args.agents, "turns": args.turns,
        "tool_tokens": args.tool_tokens, "system_tokens": args.system_tokens,
        "wall_s": round(wall, 1), "errors": len(results) - len(ok),
        "ttft_median_s": med([r["ttft"] for r in later]),
        "ttft_per_1k_prompt_ms": med([1000 * r["ttft"] / (r["prompt_tokens"] / 1000) / 1000
                                      for r in later if r["prompt_tokens"]]),
        "decode_tps_median": med([r["decode_tps"] for r in ok if r["decode_tps"]]),
        "recall_accuracy": round(sum(r["correct"] for r in checks) / len(checks), 3) if checks else None,
        "per_turn": [{"turn": t, "prompt_tokens": med([r["prompt_tokens"] for r in rs]),
                      "ttft_s": med([r["ttft"] for r in rs])} for t, rs in sorted(by_turn.items())],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--agents", type=int, default=4, help="parallel simulated agents")
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--tool-tokens", type=int, default=3000, help="size of each tool result")
    p.add_argument("--system-tokens", type=int, default=12000, help="Claude Code sends ~12-20k")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--label", default="run")
    p.add_argument("--out", help="write the full results as JSON")
    args = p.parse_args()

    results, t0 = [], time.time()
    threads = [threading.Thread(target=agent, args=(i, args, results)) for i in range(args.agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = summarize(results, time.time() - t0, args)
    for row in summary["per_turn"]:
        print("turn %2d  prompt %7s tok  ttft %6.2f s" % (row["turn"], row["prompt_tokens"], row["ttft_s"]))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_turn"}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
