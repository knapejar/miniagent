# miniagent

**A terminal coding agent for local models.** It runs your shell, edits your files,
searches and browses the web, and keeps working across long tasks — driven by a model
running on your own machine, with no API key and nothing leaving the computer.

Built and tuned against [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B)
in [LM Studio](https://lmstudio.ai/) on Windows 11. It speaks the ordinary
OpenAI-compatible API, so llama.cpp, Ollama, vLLM, SGLang or a remote endpoint work
just as well.

It also drives **Qwen3.8-27B on a free Kaggle TPU**, served by
[`kaggle-tpu-lab/`](kaggle-tpu-lab/) in this repository: `python miniagent.py --kaggle`.
See [Qwen3.8 on a Kaggle TPU](#qwen38-on-a-kaggle-tpu).

## What is where

| Path | What |
|---|---|
| `miniagent.py`, `agent/`, `ui/`, `hints/`, `tests/` | the agent (runs on your PC, standard library only) — [Layout](#layout) |
| `tasks/`, `verify/` | benchmark tasks and their checkers |
| `searxng/` | settings for a local SearXNG (web search) |
| [`kaggle-tpu-lab/`](kaggle-tpu-lab/) | the model side on a free Kaggle TPU: **Qwen3.8-27B** (vllm-tpu, `qwen38-27b/`) and **GLM-5.3-Flash** (own JAX engine, `glm53-flash/`); `launch.py serve [--model glm53-flash]` / proxy / cmd / ssh / stop, the shared control plane `common/ktl_control.py`, [operations runbook](kaggle-tpu-lab/docs/operations.md). Merged with its history from `knapejar/kaggle-tpu-lab` and upstream [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) |
| `miniagent-kaggle.cmd`, `kaggle-tpu-lab/qwen38-27b/tools/claude-qwen.cmd` | shortcuts: miniagent / Claude Code on the Kaggle model |

The same kernels also run under **DeployMan** (a separate local tool for launching services
on free-tier machines) as `services/qwen38-api/` and `services/glm53-api/`: a copy of the
kernel built with `python kaggle-tpu-lab/tools/build_kernel.py <model> <file>` plus a shared
`run.py`, with a queue-time history (`deployman stats`) and an SSH tunnel that is up from the
kernel's first seconds. Change a kernel here, then rebuild it there.

```
  ▐▛███▜▌   miniagent 0.2.0
 ▝▜█████▛▘  an agent for local models

  model    spark-x2.5-4b   @ http://127.0.0.1:1234/v1
  context  65.5k   crop at 70%   reasoning: none   max steps: 60
  cwd      C:\projects\demo
  tools    ask, browse, edit, grep, plan, read, sh, websearch, write   approval: auto

› find the GitHub issue about spark2_5 support and tell me who opened it and when

● websearch(lmstudio spark2_5 issue)
  ⎿  via searxng
     1. Feature request: update the llama.cpp engine to support spark2_5 ...
● browse(https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2378  [outline])
  ⎿  Feature request: ... - Issue #2378 - lmstudio-ai/lmstudio-bug-tracker
     1805 chars | 1 parts | 14 headings | 9 links

Issue #2378, opened by wszbdy on 7 September 2026.

──────────────────────────────────────────────────────────────
  done   context 2.9k/65.5k ░░░░░░░░ 4%  ·  8.3k tok  ·  29.5 tok/s  ·  4 steps  ·  12s
```

## Requirements

- **Windows 11** (Windows 10 works too; the shell tool targets `cmd.exe` and PowerShell)
- **Python 3.8+** — no third-party packages, standard library only
- A local model server. [LM Studio](https://lmstudio.ai/) is the shortest path.
- Optional: **Docker**, for a local SearXNG that makes web search reliable

## Install and run

```bat
git clone https://github.com/knapejar/miniagent.git
cd miniagent

:: start a model
lms server start --port 1234
lms load spark-x2.5-4b --context-length 65536 -y

python miniagent.py                          :: interactive
python miniagent.py "delete the old logs"    :: one shot
python miniagent.py -f task.txt -C C:\work   :: a task from a file, in another directory
```

Point it somewhere else with flags or environment variables:

```bat
python miniagent.py --url http://127.0.0.1:1234/v1 --model my-model --ctx 32768

set MINIAGENT_URL=http://192.168.1.10:1234/v1
set MINIAGENT_MODEL=qwen3-coder-30b
```

### Web search (optional, recommended)

Search engines block automated queries from a home address within a handful of
requests. A SearXNG running on your own machine asks around twenty engines on your
behalf and rotates them, so no single engine ever sees a burst:

```bat
python -c "import secrets; print(secrets.token_hex(32))"
:: put that value into searxng/settings.yml as secret_key, then:

docker run -d --name searxng --restart unless-stopped -p 127.0.0.1:8080:8080 ^
  -v %CD%/searxng:/etc/searxng searxng/searxng:latest
```

`searxng/settings.yml` ships with the JSON API enabled, the rate limiter off for local
use and a wide engine list. miniagent finds it automatically on `127.0.0.1:8080`;
`MINIAGENT_SEARX` points elsewhere, `MINIAGENT_SEARX=off` skips it. Without it, search
falls back to direct engines, and `BRAVE_API_KEY` or `TAVILY_API_KEY` are used first
when present.

The kaggle-tpu-lab proxy also listens on 8080. With `--kaggle` SearXNG is skipped
unless `MINIAGENT_SEARX` is set, so run SearXNG on another port there
(`-p 127.0.0.1:8888:8080` and `set MINIAGENT_SEARX=http://127.0.0.1:8888`).

## Qwen3.8 on a Kaggle TPU

[kaggle-tpu-lab](kaggle-tpu-lab/) serves Qwen3.8-27B (bf16,
up to 262k context, ~130 tok/s) with vLLM on Kaggle's free TPU v5e-8 and opens public
tunnels to it. miniagent runs here, on your machine — its tools touch your files and
your shell — and only the model runs on Kaggle.

```
miniagent ──► http://127.0.0.1:8080/v1 ──► tunnel (cloudflared / pinggy / ngrok) ──► front server ──► vLLM on the Kaggle TPU
               launch.py proxy
               (adds the API key, follows whichever tunnel is live)
```

### Run it

miniagent finds `launch.py` in `kaggle-tpu-lab\` of this repository (or in `KAGGLE_TPU_LAB`,
if set):

```bat
:: 1. start the model - once per session, it takes a TPU slot plus ~22 min
cd kaggle-tpu-lab
python launch.py serve            :: pushes the kernel; Ctrl-C detaches, it keeps running
python launch.py status -f        :: follow it until "YOUR ENDPOINT IS LIVE"

:: 2. start the agent
cd ..
python miniagent.py --kaggle                        :: interactive
python miniagent.py --kaggle "fix the failing test" :: one shot
python miniagent.py --kaggle --wait -f task.txt     :: wait for the kernel, then run
miniagent-kaggle.cmd                                :: the same as --kaggle

:: 3. end the session, so it stops using TPU quota
cd kaggle-tpu-lab
python launch.py stop
```

That is all the setup there is. With `--kaggle`, before the first step, miniagent:

1. reads `%USERPROFILE%\.kaggle-tpu-lab.json`, written by `launch.py serve` — the kernel,
   its ntfy progress topic and the API key;
2. reads that ntfy topic to see whether the kernel is live, still starting, or ended.
   It only reads; nothing is sent to the kernel. A kernel that is still queued or
   compiling is reported and the prompt opens anyway — type `/kaggle` once it is live,
   or start with `--wait`;
3. starts `launch.py proxy` in the background when nothing listens on
   `127.0.0.1:8080` yet (log in `runs\proxy.log`) and stops it again on exit. A proxy
   you already run yourself is simply used;
4. reads the served model name and context limit from `/v1/models`;
5. skips SearXNG, whose default port the proxy holds.

The banner then shows the backend:

```
  model    qwen3.8-27b   @ http://127.0.0.1:8080/v1
  backend  kaggle-tpu-lab   you/qwen38-tpu-serve   proxy started by miniagent
  context  131.1k   crop at 70%   reasoning: medium   max steps: 60
```

### What differs from the local profile

| | `--profile local` (default) | `--kaggle` |
|---|---|---|
| endpoint | `MINIAGENT_URL`, `http://127.0.0.1:1234/v1` | `MINIAGENT_KAGGLE_URL`, `http://127.0.0.1:8080/v1` |
| model | `spark-x2.5-4b` | `qwen3.8-27b`, confirmed from `/v1/models` |
| API key | `MINIAGENT_KEY` | from `.kaggle-tpu-lab.json` (the proxy adds it anyway) |
| tool-call format | Spark `<arg_key>` | Qwen `<function=…><parameter=…>` |
| context | 65 536 | 131 072, capped at the server's `max_model_len` |
| max tokens per step | 8 192 | 16 384 (room for reasoning) |
| sampling | temperature 1.0, top_p 0.95 | not sent — the model's own generation config |
| reasoning | `reasoning_effort`, off | `chat_template_kwargs`, medium |
| stop on `</tool_call>` | server side | client side, on the answer only |
| retries | none | 3, on tunnel errors (502/503/504/52x/530) and refused connections |

Why each one:

- **Tool-call format.** Qwen3.8 was trained on the `qwen3_coder` XML, not Spark's. vLLM
  parses that format only when a request carries `tools`, and miniagent never sends
  them (see `agent/protocol.py`), so the system prompt describes the format the way
  Qwen's chat template does and miniagent parses it. Values keep their indentation,
  and tool results are not wrapped in `<tool_response>` twice, because the template
  already wraps them. The parser reads both formats, so `--dialect` only changes the prompt.
- **Reasoning.** Qwen3.8 has the levels `low`, `medium` and `xhigh`, set through
  `chat_template_kwargs`. The top-level `reasoning_effort` is validated by vLLM
  against OpenAI's values (there is no `xhigh`) and the template does not read it, so
  it is not sent. `--effort none` turns thinking off, `high` (or `xhigh`) is
  the top level, and `/effort` changes it mid-session.
- **Stop sequence.** The model thinks before it calls a tool. A server-side stop on
  `</tool_call>` would also fire when it merely mentions the tag while reasoning. So
  miniagent watches the answer instead: after a complete call it lets the turn end
  normally, which keeps the token counts, and it abandons the stream when the model
  starts writing the tool's result itself.
- **Context 131k.** vllm-tpu 0.28 has no prefix caching for this model, so each step
  re-reads the whole prompt at ~10k tok/s. At 131k a step stays under ~10 s. `--ctx 262144`
  uses the full window.
- **Retries.** Pinggy URLs rotate hourly. While the proxy re-resolves a tunnel it
  answers 502, and a step waits 10, 20 and 30 s before it gives up.

### Kaggle options and settings

```
--kaggle                 same as --profile kaggle (or set MINIAGENT_PROFILE=kaggle)
--wait                   wait until the kernel is live and the endpoint answers
--no-proxy               never start launch.py proxy; run it yourself
--url URL                a tunnel URL directly, e.g. https://xxxx.trycloudflare.com/v1
--dialect spark|qwen     override the tool-call format
--effort LEVEL           none | low | medium | high | xhigh

KAGGLE_TPU_LAB           kaggle-tpu-lab directory (or its launch.py)
KAGGLE_TPU_LAB_STATE     launch state file, default %USERPROFILE%\.kaggle-tpu-lab.json
MINIAGENT_KAGGLE_URL     endpoint, default http://127.0.0.1:8080/v1
MINIAGENT_KAGGLE_MODEL   model name, default qwen3.8-27b
MINIAGENT_KAGGLE_KEY     API key, default: from the state file
```

### When something does not work

| symptom | what to do |
|---|---|
| `the Kaggle kernel is not live yet` | it is queued for a TPU or still compiling: `python launch.py status -f`, then `/kaggle` |
| `the Kaggle kernel has ended` | `python launch.py serve` starts a new one, then restart miniagent: a proxy it started still follows the old launch |
| `the proxy did not start` | the last lines of `runs\proxy.log` are printed. Port taken? Use `--url http://127.0.0.1:8081/v1` |
| `HTTP 502: kaggle-tpu-lab proxy: no live tunnel` | no tunnel answers. `python launch.py cmd tunnel` probes them, `python launch.py cmd tunnel pinggy` restarts pinggy |
| the API answers `503 server_unavailable` | vLLM is compiling, restarting or crashed; the kernel is still reachable: `python launch.py cmd status`, `cmd log 120`, `python launch.py ssh` — see [the runbook](kaggle-tpu-lab/docs/operations.md#4-recovery-runbook) |
| a step ends with `no tool call (finish=length …)` | the reasoning used up the budget: `/effort low` or `--max-tokens 32768` |

The endpoint is public and protected only by the API key. miniagent masks the key in
everything it prints and traces, and never shows it to the model.

## Tools

| tool | what it does |
|---|---|
| `sh` | Runs a command. `cmd.exe` by default; PowerShell syntax is detected and routed there. Multi-line input runs as a script, and the working directory persists between calls. |
| `read` | Reads a text file with line numbers, or a slice of one. |
| `write` | Creates or overwrites a file. Python and JSON are syntax-checked immediately. |
| `edit` | Replaces an exact string — the simplest reliable patch for a small model. |
| `grep` | Regex search across files, with glob and subtree filters. |
| `websearch` | Searches the web. One query per line runs them all in parallel. |
| `browse` | Opens a page: `outline` for the heading tree and links, `text` with `part=N` to read it in pieces, `links` for every absolute URL, `find=REGEX` to locate something inside it. One URL per line opens several at once. |
| `plan` | The agent's checklist, pinned in context and never dropped by cropping. |
| `ask` | Stops and asks you when it is blocked by something it cannot do itself. |

## Features

**Browsing, not just fetching.** An outline first — title, size, heading tree, links —
then the page in addressable parts, or a regex straight to the line you want. Relative
links are resolved to absolute URLs, and `<main>`/`<article>` is preferred over site
chrome, so Wikipedia arrives as prose and a GitHub issue as the issue.

**Context managed by the agent.** At 70% of the window the middle is cut away; the
system prompt, the original task and the plan always survive. The length estimate
calibrates itself against the server's own token counts.

**Recovery hints.** When a tool fails in a way that has a known fix, the agent is told
the fix. `hints/` holds one JSON file per tool — `{"when": regex, "say": text}` — so
teaching it a new recovery is one entry rather than a code change. `MINIAGENT_HINTS`
merges in a directory of your own.

**Guards for long runs.** An empty or truncated reply is never mistaken for a finished
task. A call that fails three times the same way is blocked. Progress is measured as
novelty, so browsing eleven different pages is left alone while three calls that bring
back nothing new get a nudge.

**Live metrics.** Tokens, tokens per second and elapsed time while it generates; a
summary of steps, token split, time in tools versus generation, context crops and
repeat loops when it finishes. Every run is written to a JSONL trace under
`~/.miniagent/projects/<slug>/runs/`, which makes two models comparable on the same task.

**Memory and history.** Nothing is written into the directory the agent works in.
Per project, its state lives in `~/.miniagent/projects/<slug>/`, where the slug is the
project's absolute path with the separators flattened to dashes
(`C--Users-Jarda-REPO-Moje-miniagent`). `memory/INDEX.md` holds the notes the agent
writes for itself and is loaded into the system prompt on every start; `runs/*.jsonl`
holds every past trace, so the agent can grep its own history of this project.
Set `MINIAGENT_HOME` to move the whole thing elsewhere.

The notes do not appear by themselves: once the answer is delivered, a short
**memory pass** runs, with the single job of updating `INDEX.md` and only the file
tools to do it with. The current index is handed to it in the prompt, so it writes
straight back instead of spending a round trip reading it.

It runs **in the background**: the prompt comes up the moment the answer is
finished and you type into it as usual, while the pass works behind it. Nothing
is printed over the line you are typing on - the result lands above the next
prompt as one collapsed line, and `/memory` shows what it wrote. Submitting the
next task cuts the pass short if it is still going. Its own messages are dropped
from the history afterwards. It costs one model call per finished task, and
`--no-memory` turns it off.

**Rendered answers.** Bold, italic, headings, lists, quotes, code fences, aligned
tables and highlighted links, streamed as they arrive.

**Safety.** Ordinary commands run unattended; risky ones ask first. Credentials are
masked before they can reach the transcript, the trace or the model's context, and
secrets held in environment variables are offered to the model by name only. `esc`
stops the agent mid-token.

## Interactive commands

```
/help   /plan    /memory  /stats   /tools   /hints   /context   /cwd
/effort /approve /steps   /clear   /trace   /kaggle  /quit

!<command>    run a shell command directly, without the model
esc           stop the agent immediately
```

## Options

```
-f, --task-file FILE     read the task from a file
-C, --workdir DIR        working directory for the agent
    --kaggle             Qwen3.8-27B on a Kaggle TPU, see above
    --profile NAME       local | kaggle
    --url URL            OpenAI-compatible endpoint
    --model NAME         model identifier
    --ctx N              context window (default 65536)
    --max-steps N        step limit per task (default 60)
    --max-tokens N       generation limit per step (default 8192)
    --dialect NAME       auto | spark | qwen (tool-call format)
    --effort LEVEL       default | none | low | medium | high | xhigh
    --temperature F      default 1.0
    --approve MODE       auto | all | none
    --crop-at F          share of the context that triggers cropping
    --remind-every N     steps between goal reminders
    --no-stream          disable streaming
    --no-color           plain output
    --no-redact          do not mask secrets in tool output
    --no-reasoning       do not print the model's reasoning (shown live by default)
    --run-name NAME      name of the trace file
```

## Layout

```
miniagent.py          entry point, REPL, slash commands
agent/
  config.py           settings, arguments, the local and kaggle profiles
  protocol.py         tool-call formats (spark, qwen), system prompt
  kaggle.py           Qwen3.8 on a Kaggle TPU: launch state, proxy, readiness
  llm.py              OpenAI-compatible client, streaming, cancellation
  session.py          history, context cropping, pinned plan
  loop.py             the agent loop, a generator of events
  guards.py           failure memory, repeat and novelty detection
  hints.py            per-tool recovery hints
  jobs.py             parallel fan-out and background jobs
  web.py              search providers and HTTP
  page.py             outline, absolute links, addressable parts
  secrets.py          credential masking
  metrics.py          tokens, tokens per second, timings
  trace.py            JSONL run trace
  paths.py            where state lives: ~/.miniagent/projects/<slug>/
  tools/              sh, read, write, edit, grep, websearch, browse, plan, ask
ui/
  ansi.py             colours, spinner, formatting
  keys.py             esc watcher
  markdown.py         renders the model's markdown
  render.py           draws loop events
hints/                one JSON file per tool
tasks/                benchmark tasks, including ten hard ones
verify/               independent checkers for those tasks
tests/                171 tests
kaggle-tpu-lab/       the model side on Kaggle (own README)
```

The loop is a generator of events and prints nothing itself, so the interface can be
replaced or switched off without touching the logic.

```bat
python -m unittest discover -s tests
```

## Comparing models

Same harness, same protocol, a different endpoint — so the difference is the model:

```bat
python miniagent.py -f tasks/hard/h08_markdown.txt --run-name local_h08
python miniagent.py -f tasks/hard/h08_markdown.txt --run-name cloud_h08 ^
       --url https://api.anthropic.com/v1 --model claude-sonnet-5 --api-key %API_KEY%
```

Compare the traces in `~/.miniagent/projects/<slug>/runs/` — steps, tokens, crops,
loops — and above all the independent
checkers in `verify/`. An agent's own claim of success is not evidence: during testing
one model wrote a test file with 108 asserts, never called a single test function,
printed `ALL OK` and reported the task complete.

## Licence

MIT — see [LICENSE](LICENSE).
