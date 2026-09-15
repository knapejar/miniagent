# miniagent

**A terminal coding agent for local models.** It runs your shell, edits your files,
searches and browses the web, and keeps working across long tasks — driven by a model
running on your own machine, with no API key and nothing leaving the computer.

Built and tuned against [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B)
in [LM Studio](https://lmstudio.ai/) on Windows 11. It speaks the ordinary
OpenAI-compatible API, so llama.cpp, Ollama, vLLM, SGLang or a remote endpoint work
just as well.

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

Tool calls come in three dialects (`--dialect`, `MINIAGENT_DIALECT`). `spark` and
`nanbeige` are the text formats those small local models were trained on; every other
model gets `openai` — standard function calling (`tools` in the request, structured
`tool_calls` back, results as `role: tool`), which vLLM, SGLang, LM Studio and hosted
APIs parse on the server. A remote vLLM endpoint with a key:

```bat
python miniagent.py --url https://xxxx.trycloudflare.com/v1 --api-key sk-... --model qwen3.8-27b --effort low
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
`MINIAGENT_SEARX` points elsewhere. Without it, search falls back to direct engines,
and `BRAVE_API_KEY` or `TAVILY_API_KEY` are used first when present.

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
repeat loops when it finishes. Every run is written to `runs/*.jsonl`, which makes two
models comparable on the same task.

**Rendered answers.** Bold, italic, headings, lists, quotes, code fences, aligned
tables and highlighted links, streamed as they arrive.

**Safety.** Ordinary commands run unattended; risky ones ask first. Credentials are
masked before they can reach the transcript, the trace or the model's context, and
secrets held in environment variables are offered to the model by name only. `esc`
stops the agent mid-token.

## Interactive commands

```
/help   /plan    /stats   /tools   /hints   /context   /cwd
/effort /approve /steps   /clear   /trace   /quit

!<command>    run a shell command directly, without the model
esc           stop the agent immediately
```

## Options

```
-f, --task-file FILE     read the task from a file
-C, --workdir DIR        working directory for the agent
    --url URL            OpenAI-compatible endpoint
    --model NAME         model identifier
    --ctx N              context window (default 65536)
    --max-steps N        step limit per task (default 60)
    --max-tokens N       generation limit per step (default 8192)
    --effort LEVEL       default | none | low | medium | high
    --temperature F      default 1.0
    --approve MODE       auto | all | none
    --crop-at F          share of the context that triggers cropping
    --remind-every N     steps between goal reminders
    --no-stream          disable streaming
    --no-color           plain output
    --no-redact          do not mask secrets in tool output
    --show-reasoning     print the model's reasoning
    --run-name NAME      name of the trace file
```

## Layout

```
miniagent.py          entry point, REPL, slash commands
agent/
  config.py           settings and arguments
  protocol.py         tool-call format, system prompt
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
  tools/              sh, read, write, edit, grep, websearch, browse, plan, ask
ui/
  ansi.py             colours, spinner, formatting
  keys.py             esc watcher
  markdown.py         renders the model's markdown
  render.py           draws loop events
hints/                one JSON file per tool
tasks/                benchmark tasks, including ten hard ones
verify/               independent checkers for those tasks
tests/                135 tests
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

Compare `runs/*.jsonl` — steps, tokens, crops, loops — and above all the independent
checkers in `verify/`. An agent's own claim of success is not evidence: during testing
one model wrote a test file with 108 asserts, never called a single test function,
printed `ALL OK` and reported the task complete.

## Licence

MIT — see [LICENSE](LICENSE).
