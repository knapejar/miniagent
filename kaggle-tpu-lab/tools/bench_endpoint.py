"""Speed of a Qwen3.8 endpoint: TTFT, decode tok/s, prefill, 4-stream concurrency.

    python tools/bench_endpoint.py [base_url] [out.json]

Default base_url is the local proxy (http://127.0.0.1:8080/v1). For a raw tunnel set
KTL_API_KEY to the launch's API key."""
import os
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1"
MODEL = "qwen3.8-27b"


def stream(messages, max_tokens, kwargs=None, timeout=1800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("KTL_API_KEY", "x")})
    t0 = time.time()
    first = None
    usage = {}
    text, reasoning = [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                if (piece or rpiece) and first is None:
                    first = time.time()
                text.append(piece)
                reasoning.append(rpiece)
    end = time.time()
    ct = usage.get("completion_tokens", 0)
    ttft = (first or end) - t0
    decode_s = max(1e-6, end - (first or end))
    return {"ttft_s": round(ttft, 2), "total_s": round(end - t0, 2),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": ct,
            "decode_tok_s": round((ct - 1) / decode_s, 1) if ct > 1 else None,
            "text": "".join(text), "reasoning_chars": len("".join(reasoning))}


def show(name, r):
    print("%-34s ttft %6.2fs  total %6.1fs  prompt %6s  gen %5s  decode %6s tok/s" % (
        name, r["ttft_s"], r["total_s"], r["prompt_tokens"], r["completion_tokens"],
        r["decode_tok_s"]), flush=True)


results = {}
off = {"enable_thinking": False}

# warm-up
show("warmup", stream([{"role": "user", "content": "Say hi."}], 16, off))

r = stream([{"role": "user", "content": "Write a detailed 600-word essay about the history of the printing press."}], 800, off)
show("decode 800 tok, thinking off", r)
results["decode_single"] = {k: v for k, v in r.items() if k != "text"}

r = stream([{"role": "user", "content": "Write a Python function that checks whether a string is a valid IPv4 address, with a few asserts."}], 4000, {"reasoning_effort": "low"})
show("coding q, reasoning low", r)
results["coding_low"] = {k: v for k, v in r.items() if k != "text"}

filler = ("The quick brown fox jumps over the lazy dog while the committee reviews "
          "section %d of the annual infrastructure report in considerable detail. ")
for n_words in (2000, 20000):
    doc = "".join(filler % i for i in range(n_words // 20))
    r = stream([{"role": "user", "content": doc + "\n\nHow many sections were mentioned? Answer with one number."}], 16, off)
    show("prefill ~%d words" % n_words, r)
    results["prefill_%d" % n_words] = {k: v for k, v in r.items() if k != "text"}
    if r["prompt_tokens"]:
        results["prefill_%d" % n_words]["prefill_tok_s"] = round(r["prompt_tokens"] / r["ttft_s"], 0)
        print("    prefill ~%.0f tok/s" % (r["prompt_tokens"] / r["ttft_s"]), flush=True)

for conc in (4,):
    outs = [None] * conc

    def run(i):
        outs[i] = stream([{"role": "user", "content": "Write a 400-word story about robot number %d." % i}], 500, off)
    t0 = time.time()
    ths = [threading.Thread(target=run, args=(i,)) for i in range(conc)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    tot = sum(o["completion_tokens"] for o in outs)
    per = [o["decode_tok_s"] for o in outs]
    print("%d concurrent: %d tokens in %.1fs = %.0f tok/s aggregate, per-stream %s" % (
        conc, tot, wall, tot / wall, per), flush=True)
    results["concurrent_%d" % conc] = {"tokens": tot, "wall_s": round(wall, 1),
                                        "aggregate_tok_s": round(tot / wall), "per_stream": per}

json.dump(results, open(sys.argv[2] if len(sys.argv) > 2 else "rawbench.json", "w"), indent=1)
print("saved", flush=True)
