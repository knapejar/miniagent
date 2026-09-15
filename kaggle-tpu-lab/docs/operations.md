# Operations: control channels, SSH, recovery, incidents

What you need to run Qwen3.8-27B on a Kaggle TPU without losing control of the session,
and what went wrong on 2026-09-15 that shaped the current design.

## 1. Architecture

```
your PC                                   Kaggle kernel (TPU v5e-8 or CPU nettest)
─────────                                 ────────────────────────────────────────────
launch.py proxy :8080 ──┐                 ┌─ front server 127.0.0.1:8000 (in serve_qwen38.py)
miniagent / Claude Code ─┤  HTTPS 443     │    /ktl/health  /ktl/tunnels  /ktl/cmd
launch.py cmd / shell ───┼──► ngrok ──────┤    everything else ─► vLLM 127.0.0.1:8001
                         ├──► cloudflared ┤
                         └──► pinggy ─────┘
launch.py ssh ───────────┬──► cloudflared-ssh ─► sshd 127.0.0.1:2222 (root, your key only)
                         └──► pinggy-ssh (tcp)─┘
launch.py (fallback) ────────► ntfy.sh topics ◄──► command listener + progress events
```

* Tunnels, sshd and the front server start in the first seconds of the kernel, before the
  Python runtime installs. Any failure later still leaves the kernel reachable.
* The front server lives in the kernel script itself, so it keeps answering while vLLM
  compiles, after it crashed, and while it restarts. API calls during that time get
  `503 {"error": {"type": "server_unavailable", ...}}` instead of a tunnel error page.
* `~/.kaggle-tpu-lab.json` (written by `serve`/`nettest`) holds the kernel id, the ntfy
  topics, the API key and the last URL of every tunnel. `~/.kaggle-tpu-lab/id_ed25519` is the
  SSH key; `~/.kaggle-tpu-lab/cloudflared.exe` is downloaded on the first `launch.py ssh`.

## 2. Everyday commands

```bash
python launch.py nettest                 # CPU kernel, no TPU queue: tunnels + control + SSH, ~5 min
python launch.py serve                   # TPU kernel; watch progress, Ctrl-C only detaches
python launch.py status -f               # re-attach to the progress stream
python launch.py proxy                   # http://127.0.0.1:8080/v1 for every local client
python launch.py cmd status              # server + tunnels
python launch.py cmd log 80              # tail of vllm.log (every vLLM line)
python launch.py cmd sh "ps aux | grep vllm"
python launch.py ssh                     # interactive root shell in the kernel
python launch.py ssh --via pinggy "tail -f /kaggle/working/vllm.log"
python launch.py stop                    # graceful stop, then delete the kernel (ends the session)
```

Run the nettest and the serve launch from **different home directories** if they must run
at the same time: both write `~/.kaggle-tpu-lab.json`, and `stop` deletes whichever kernel
that file names.

```bash
# second, isolated launcher state (Git Bash); copy the Kaggle token into it first
mkdir -p /tmp/ktl-home/.kaggle && cp ~/.kaggle/access_token /tmp/ktl-home/.kaggle/
USERPROFILE=$(cygpath -w /tmp/ktl-home) HOME=/tmp/ktl-home KAGGLE_USERNAME=<you> python launch.py nettest
```

## 3. Clients

| Client | Point it at | Notes |
|---|---|---|
| anything OpenAI-compatible | `http://127.0.0.1:8080/v1` (proxy), any key | or a tunnel URL + the API key from the banner |
| Claude Code | `tools\claude-qwen.cmd` (proxy on :8080) | the proxy maps `output_config.effort` `high -> medium`, `max -> xhigh`; a raw tunnel returns HTTP 400 for `high` |
| miniagent | `python miniagent.py --kaggle` | needs `integrations/miniagent-vllm-tool-calls.patch` (see below) |
| another PC | tunnel URL + API key, or copy `~/.kaggle-tpu-lab.json` there and run `launch.py proxy` | pinggy URLs rotate hourly; cloudflared URLs live until the kernel restarts its tunnel |

**miniagent patch.** vLLM runs with `--enable-auto-tool-choice --tool-call-parser
qwen3_coder`, which extracts `<tool_call>` markup from the answer **even when the request
lists no tools** and streams it as structured `delta.tool_calls` with the text removed.
miniagent read only text, saw "no tool call" three times and stalled. The patch rebuilds the
markup from `tool_calls` (streaming and non-streaming) and adds four tests. Apply it in the
miniagent checkout (made against commit `2d78b91`):

```bash
cd ../miniagent && git apply ../kaggle-tpu-lab/integrations/miniagent-vllm-tool-calls.patch
python -m unittest discover -s tests
```

**Speed seen by agents.** miniagent reports tokens / whole step time. Without prefix
caching every step re-reads the whole conversation (~9-10k tok/s prefill), so a 60-token
step on a 16k context shows ~25 tok/s although decode runs at ~100 tok/s. Lower
`/effort`, a smaller `--ctx`, and one client at a time help; prefix caching needs a newer
vllm-tpu.

`tools/bench_endpoint.py [base_url]` measures TTFT, decode, prefill and 4-stream
concurrency through any endpoint. Numbers from 2026-09-15 through the proxy and a
cloudflared tunnel:

| Test | Result |
|---|---|
| decode, 800 tokens, thinking off | 95 tok/s |
| coding answer, reasoning low | 136 tok/s |
| prefill 2.6k / 26.9k tokens | 0.65 s / 2.9 s TTFT (~9.2k tok/s) |
| 4 concurrent streams | 270 tok/s aggregate, 83-96 each |

## 4. Recovery runbook

**The API answers 503 `server_unavailable`.** vLLM is compiling, restarting or dead.
`python launch.py cmd status` shows the phase. While serving, a crash triggers an in-place
restart (`server-restart` event, ~20 min, up to `server_restarts`, default 3). Find the
cause with `python launch.py cmd log 120` or `launch.py ssh` then
`grep -n "EngineCore.*ERROR" /kaggle/working/vllm.log | head`.

**`cmd` answers "no answer".** The launcher tried every stored tunnel and then ntfy.
1. `python launch.py ssh` (independent of the front server and of ntfy).
2. Tunnel URLs changed? `python launch.py status` re-reads ntfy events and stores new URLs.
3. Test a URL by hand: `curl https://<tunnel>/ktl/health`.

**A tunnel is down.** `python launch.py cmd tunnel restart` (watchdog restarts within
~30 s), `cmd tunnel pinggy` to start pinggy, `cmd tunnel ngrok <url> <token>` to add ngrok.

**Restart vLLM by hand inside the session** (keeps the TPU slot):
```bash
python launch.py ssh
# inside the kernel
pkill -f vllm.entrypoints.openai.api_server; sleep 15
grep -c "KTL fix v2" /tmp/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
# the kernel's keep-alive loop notices the exit and restarts vLLM itself
```

**Keep a session whose kernel script is about to exit.** The Kaggle session ends when
`/kaggle/src/script.py` (PID 7) exits. `kill -STOP 7` freezes it and the session stays up
until Kaggle's 9-hour cap; a replacement supervisor must then be started with `setsid
nohup`. Use this only as a last resort: the frozen process's own channels freeze with it.

**Give up.** `python launch.py stop`, or `python -m kaggle kernels delete <user>/<slug>`
(Kaggle API, independent of every channel). Deleting the kernel ends the session and stops
quota use.

## 5. Incident log, 2026-09-15

| Time (CEST) | What happened |
|---|---|
| earlier attempt | cloudflared quick tunnel never connected (port 7844 blocked on the network) and there was no other way into the kernel |
| 13:54 | TPU kernel queued; added pinggy (443) + ntfy command channel + local proxy, verified on a CPU nettest |
| 18:04 | TPU live after 4 h 10 min in the queue |
| 18:05 | miniagent stalled: vLLM returned tool calls as structured `tool_calls`, miniagent read text only → client patch |
| 18:19 | Claude Code: HTTP 400 `Unexpected reasoning effort high` → effort mapping in the proxy |
| 18:25 | **crash 1**: `AttributeError: __delitem__` in `scheduler.update_draft_token_ids_in_output` (MTP + 3 concurrent clients). The kernel held the session 30 min before exiting. Rescued over ntfy: patched the scheduler, started a copy of the script reusing the venv, froze PID 7 |
| 18:49 | vLLM back (restart took ~20 min with cached graphs) |
| 19:03 | **crash 2**: `BufferError: to_dlpack ... Sharding over 8 devices` in xgrammar `validate_tokens` — a structured-output request + MTP drafts as a sharded JAX array |
| 19:06 | ntfy stopped delivering to/from the kernel (most likely rate limiting of the kernel's IP after hundreds of messages). Tunnels were up, but only vLLM listened behind them → no way in. Session deleted at ~19:20 (2.14 TPU h used) |
| 19:27-19:31 | new design verified on a Kaggle CPU kernel (section 6) |

### Lessons

1. **A tunnel must lead to something that is always there.** Tunnels pointed straight at
   vLLM, so a vLLM crash turned working tunnels into dead ends. Now they point at the front
   server with control routes.
2. **Never rely on one free shared service for control.** ntfy.sh is the fallback now;
   HTTPS over tunnels and SSH are primary.
3. **Open channels before anything that can fail.** Tunnels used to start only at step 5.
4. **Crash → restart in place, not hold → exit.** A new kernel costs hours of queue.
5. **Test with real concurrency and real clients before the long queue.** A fake vLLM
   server hid both the tool-call format and the effort validation; the MTP crashes appeared
   only with several real agents.
6. **Keep the message budget small.** Progress events are deduplicated; command output goes
   over HTTPS when a tunnel is up.

## 6. Verification (2026-09-15)

On a Kaggle CPU kernel (`launch.py nettest`), from a Windows 11 PC whose own network blocks
port 7844:

| Check | Result |
|---|---|
| cloudflared, pinggy API tunnels | live |
| cloudflared-ssh, pinggy-ssh | online |
| `launch.py cmd status` over a tunnel with ntfy disabled in the state file | OK, 0.67 s |
| kill the server behind the front, then `cmd status` | OK; API returns 503 JSON |
| `launch.py ssh --via cloudflared` (Git Bash ssh and Windows OpenSSH) | root shell, OK |
| `launch.py ssh --via pinggy` (Git Bash ssh and Windows OpenSSH) | root shell, OK |
| `launch.py stop` over the tunnel | kernel stopped and deleted |

Locally in Docker (python:3.12-slim): the same checks plus `launch.py proxy` through the
front server and the automatic `openssh-server` install.

**Not verified yet:** a full TPU `serve` with the new kernel (front server in front of real
vLLM streaming, the in-place restart loop on a real crash, and the scheduler fix v2 under
concurrent agents with structured output). Run one before relying on long sessions.
