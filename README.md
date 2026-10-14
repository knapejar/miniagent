# miniagent

**A terminal coding agent for local models.** It runs your shell, edits your files,
searches and browses the web, and keeps working across long tasks — driven by a model
running on your own machine, with no API key and nothing leaving the computer.

Built and tuned against [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B)
in [LM Studio](https://lmstudio.ai/) on Windows 11. It speaks the ordinary
OpenAI-compatible API, so llama.cpp, Ollama, vLLM, SGLang or a remote endpoint work
just as well — see [Profiles](#profiles).

miniagent is only the harness. Serving models is somebody else's job: it talks to
whatever endpoint a profile points at, and at most starts a local server for it.

| Path | What |
|---|---|
| `miniagent.py`, `agent/`, `ui/`, `hints/`, `tests/` | the agent (runs on your PC, standard library only) — [Layout](#layout) |
| `tasks/`, `verify/` | benchmark tasks and their checkers |
| `searxng/` | settings for a local SearXNG (web search) |

```
  ▐▛███▜▌   miniagent 0.2.0
 ▝▜█████▛▘  an agent for local models

  model    spark-x2.5-4b   local @ http://127.0.0.1:1234/v1
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

- **Windows 11** (Windows 10 works too; the shell tool targets `cmd.exe` and PowerShell).
  Linux and macOS work as well - there the shell tool runs bash.
- **Python 3.8+** — no third-party packages, standard library only
- A local model server. [LM Studio](https://lmstudio.ai/) is the shortest path.
- Optional: **Docker**, for a local SearXNG that makes web search reliable

## Install and run

```bat
git clone https://github.com/knapejar/miniagent.git
cd miniagent

python miniagent.py                          :: interactive, pick the model with /model
python miniagent.py --profile local          :: interactive on Spark-X2.5-4B right away
python miniagent.py "delete the old logs"    :: one shot (on local unless --profile says otherwise)
python miniagent.py -f task.txt -C C:\work   :: a task from a file, in another directory
```

Interactive miniagent starts with **no model picked** — unless `--profile` names one
or a default was stored. `/model` lists every profile with its state and switches to
the one you pick; picking `local` starts the LM Studio server and loads
Spark-X2.5-4B (`lms server start`, `lms load`) when they are not up yet:

```
› /model
  model  ·  ↑↓ enter esc
› lmstudio  local                spark-x2.5-4b            127.0.0.1:1234         starts LM Studio on pick
  lmstudio  gpubox               qwen3-coder-30b          192.168.1.10:1234      no answer
  hosted    big                  some-model-70b           api.example.com        ready
```

`ready` answers now; `starts … on pick` / `loads on pick` is brought up when chosen;
anything else (`no answer`, `no API key …`, `missing …`) says what is in the way.
A task typed before any model is picked opens the same list first.

## Profiles

A profile is one model on one backend. miniagent ships exactly one, **`local`**:
Spark-X2.5-4B in LM Studio on `127.0.0.1:1234` (`MINIAGENT_URL`, `MINIAGENT_MODEL`
and `MINIAGENT_KEY` override its endpoint, model and key).

Everything else is yours, kept out of this repository: one JSON file per provider in
`~/.miniagent/profiles/` (a folder link to a private repository of profiles works
too). The file's name is the provider, its top-level keys are shared by all of its
profiles, and a profile's own keys win:

```json
~/.miniagent/profiles/hosted.json
{
  "url": "https://api.example.com/v1",
  "api_key_env": "EXAMPLE_API_KEY",
  "headers": {"X-Data-Retention": "none"},
  "ctx": 131072,
  "retries": 2,
  "profiles": {
    "big":   {"model": "some-model-70b"},
    "small": {"model": "some-model-8b", "effort": "medium"}
  }
}
```

| key | default | what |
|---|---|---|
| `url`, `model` | `http://127.0.0.1:1234/v1`, — | the OpenAI-compatible endpoint and model id |
| `api_key` | — | else `~/.miniagent/<profile>.key`, else `~/.miniagent/<provider>.key` |
| `connector` | `lmstudio` | what brings the backend up, see below |
| `ctx`, `max_tokens` | 65536, 8192 | context window, generation limit per step |
| `temperature`, `top_p`, `top_k` | not sent | sampling; unset leaves it to the server |
| `effort` | `low` | `default` \| `none` \| `low` \| `medium` \| `high` \| `xhigh` |
| `effort_in_template` | `false` | send the effort as Qwen3.8 `chat_template_kwargs` (vLLM) |
| `dialect` | `auto` | tool-call format: `spark` \| `qwen` \| `native`, by default from the model name |
| `multi_call` | `auto` | tool calls one reply may carry: 8 in the `native` dialect, 1 for `spark` and `qwen` |
| `crop_at` | 0.7 | share of `ctx` that triggers cropping; a paid API may want it lower |
| `headers` | `{}` | extra HTTP headers the provider asks for |
| `retries`, `outage_wait` | 0, 0 | retries when the server cannot be reached, counted down on screen; seconds to wait out a server outage |
| `<key>_env` | | an environment variable that replaces `<key>` when set |
| `note` | | free text for you; miniagent ignores it |

Connectors (`agent/connectors/`) make the backend answer before the first step:

- **`lmstudio`** — for a local URL, starts the LM Studio server and loads the model
  with `lms`; for any other URL (a hosted API, a GPU box) it only checks the server answers.
- **`llamacpp`** — a model LM Studio cannot load, on its own llama.cpp build: `server`
  (path to `llama-server`), `gguf` and `server_args`. Starts it when nothing answers and
  leaves it running.
- **`kaggle`** — a model served on a Kaggle TPU by kaggle-tpu-lab: reads its launch
  state, starts its local proxy, takes the model name and context limit from
  `/v1/models`. Options: `--wait`, `--no-proxy`; `KAGGLE_TPU_LAB` points at its `launch.py`.

```bat
python miniagent.py --profile                       :: list the profiles and their state
python miniagent.py --profile big --set-default     :: remember one (MINIAGENT_PROFILE wins)
python miniagent.py --profile big --ctx 32768       :: flags win over the profile
python miniagent.py --url http://127.0.0.1:8000/v1 --model my-model   :: no profile file at all
```

## Web search (optional, recommended)

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

A model server on port 8080 (llama-server's default, the kaggle proxy) collides with it;
the kaggle connector then skips SearXNG unless `MINIAGENT_SEARX` is set. Run SearXNG on
another port in that case (`-p 127.0.0.1:8888:8080` and
`set MINIAGENT_SEARX=http://127.0.0.1:8888`).

## Tools

| tool | what it does |
|---|---|
| `sh` | Runs a command. `cmd.exe` by default; PowerShell syntax is detected and routed there. Multi-line input runs as a script, and the working directory persists between calls. |
| `read` | Reads a text file with line numbers, a slice of one, or up to four files at once. A read that stops early ends with an outline of what it did not show - its `def`/`class` lines, or the headings of a Markdown file. |
| `write` | Creates or overwrites a file. Python and JSON are syntax-checked immediately. |
| `edit` | Replaces an exact string — the simplest reliable patch for a small model. Keeps the file's line endings and encoding; `undo=true` takes its last change back. |
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

**Context managed by the agent.** At 70% of the window room is made cheapest first:
old reasoning, reads of a file that has changed since and long tool output are shrunk
in place, and only when that is not enough is the middle cut away. The system prompt,
the original task and the plan always survive. When a plan item turns DONE in a long
run, what it took to get there is dropped, so the next item starts from the task, the
plan and a list of the files already touched (`--checkpoint-at`). A reply that was all
reasoning and no answer is not kept. The length estimate calibrates itself against the
server's own token counts.

**Orientation, and a check before done.** The system prompt shows the working
directory, git branch, status and last commits, and the project's test command when
its files tell it. An answer that comes after a code file changed but before the code
ran again - or right after a failing run - is held back once, with the test command
and `git diff --stat`.

**Undo.** Every write and edit keeps what the file held before: `edit undo=true`
takes one change back, `/undo` puts back every file the last task changed, and the
first copy of each goes to `~/.miniagent/projects/<slug>/backups/`.

**Recovery hints.** When a tool fails in a way that has a known fix, the agent is told
the fix. `hints/` holds one JSON file per tool — `{"when": regex, "say": text}` — so
teaching it a new recovery is one entry rather than a code change. `MINIAGENT_HINTS`
merges in a directory of your own.

**Several calls per step, where the model can.** Every step sends the whole
conversation again, so a model that makes one call per step pays for the history
once per call - one hosted run spent 10M prompt tokens on 206 single calls. In the
`native` dialect a reply may carry up to `multi_call` calls (8 by default): the
system prompt asks the model to batch independent calls, they run in order and
their results go back together in one message. Spark and Qwen, whose format
miniagent writes into the prompt, keep one call per step and the prompt they were
tuned on. `--max-task-tokens` caps the prompt tokens of one task: past it the run
answers with what it has instead of spending on. Cached prompt tokens, when the
server reports them, are counted in the trace and the headless `result`.

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
`--no-memory` turns it off, and also leaves the memory and the pointer to old
runs out of the system prompt - for a host program that keeps its own.

**Rendered answers.** Bold, italic, headings, lists, quotes, code fences, aligned
tables and highlighted links, streamed as they arrive.

**Safety.** Ordinary commands run unattended; risky ones ask first. Credentials are
masked before they can reach the transcript, the trace or the model's context, and
secrets held in environment variables are offered to the model by name only. `esc`
stops the agent mid-token.

## Interactive commands

```
/help   /plan    /memory  /stats   /tools   /hints   /context   /cwd
/effort /approve /steps   /clear   /trace   /model   /resume  /undo   /quit

!<command>    run a shell command directly, without the model
esc           stop the agent immediately, also while it waits for the server or to retry
paste         a multi-line paste stays one [Pasted text #1 +12 lines] on the line;
              one backspace takes it out, enter sends the text itself
```

## Headless: driven by another program

`-p` runs miniagent without the interactive UI, with the same flags and the same
JSON stream as Claude Code's print mode - so a program that already drives
`claude -p` drives miniagent without changes:

```bat
python miniagent.py -p "summarize README.md"                      :: just the answer
python miniagent.py -p "summarize README.md" --output-format json :: one result object
python miniagent.py -p --input-format stream-json --output-format stream-json ^
       --include-partial-messages --session-id 6f0c...  --profile big
```

With `--input-format stream-json` the process stays up and takes one message per
stdin line until stdin closes; tasks run one after another in one conversation:

```json
{"type": "user", "message": {"role": "user", "content": "fix the failing test"}}
{"type": "control_request", "request_id": "r1", "request": {"subtype": "interrupt"}}
{"type": "control_request", "request_id": "r2", "request": {"subtype": "get_context_usage"}}
```

`interrupt` stops the running task the way esc does. Every stdout line is one
event: `system`/`init` (session id, model, tools, MCP servers), `stream_event`
(text and thinking deltas, tool_use starts - with `--include-partial-messages`),
`assistant` (text and `tool_use` blocks), `user` (`tool_result` blocks),
`system`/`compact_boundary` when the context was cropped, and a `result` that
closes each task with its answer, duration and token counts. A content of
`/compact` crops the conversation on demand.

The conversation is saved under its id in `~/.miniagent/projects/<slug>/sessions/`;
`--resume <id>` continues it in a later process, and an unknown id fails with
`No conversation found with session ID`.

| flag | what |
|---|---|
| `-p`, `--print` | headless; the task comes from the arguments, stdin or the stream |
| `--output-format` | `text` (default) \| `json` \| `stream-json` |
| `--input-format` | `text` (default) \| `stream-json` |
| `--include-partial-messages` | stream deltas as `stream_event` lines |
| `--session-id`, `--resume` | name a new conversation, continue a saved one |
| `--append-system-prompt`, `--append-system-prompt-file` | your instructions after the built-in system prompt |
| `--mcp-config` | MCP servers, `{"mcpServers": {...}}` inline or a file; repeatable |
| `--disallowedTools` | tools the model is not offered |
| `--dangerously-skip-permissions` | never ask (the default in headless mode anyway) |
| `--verbose`, `--strict-mcp-config` | accepted, nothing to change |

**MCP servers.** stdio servers (`command`, `args`, `env`) and HTTP ones (`type:
"http"`, `url`, `headers`) are started next to the agent; each of their tools is
offered as `mcp__<server>__<tool>`. `--mcp-config` is read in headless mode.

## Options

```
-f, --task-file FILE     read the task from a file
-C, --workdir DIR        working directory for the agent
    --profile [NAME]     the model to use; without a name, list them with their state
    --set-default        remember --profile for next time
    --url URL            OpenAI-compatible endpoint
    --model NAME         model identifier
    --ctx N              context window (default 65536)
    --max-steps N        step limit per task (default 60)
    --max-tokens N       generation limit per step (default 8192)
    --multi-call N       tool calls one reply may carry (default 8 native, 1 spark/qwen)
    --max-task-tokens N  prompt tokens one task may spend before it answers (0 = no limit)
    --dialect NAME       auto | spark | qwen (tool-call format)
    --effort LEVEL       default | none | low | medium | high | xhigh
    --temperature F      default 1.0
    --approve MODE       auto | all | none (default none: no confirmations)
    --crop-at F          share of the context that triggers cropping
    --keep-tail N        messages a crop leaves untouched at the end (default 12)
    --checkpoint-at F    a plan item turning DONE past this share of the context
                         drops what it took (default 0.3, 0 = never)
    --think-boost        one effort level more on the first step and after a failing run
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
  config.py           settings, arguments, profiles (local + ~/.miniagent/profiles)
  headless.py         -p: the Claude Code-compatible print mode and JSON stream
  mcp.py              MCP client: servers' tools as mcp__<server>__<tool>
  connectors/         what brings a backend up: lmstudio, llamacpp, kaggle
  protocol.py         tool-call formats (spark, qwen, native), system prompt
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
  workspace.py        what the agent is shown of the project: listing, git, test command
  tools/              sh, read, write, edit, grep, websearch, browse, plan, ask
ui/
  ansi.py             colours, spinner, formatting
  keys.py             esc watcher
  prompt.py           the prompt's line editor: pastes, history
  picker.py           arrow-key list for /model and /resume
  markdown.py         renders the model's markdown
  render.py           draws loop events
hints/                one JSON file per tool
tasks/                benchmark tasks: ten hard ones, three that edit this repository
verify/               bench.py and an independent checker per task
tests/                313 tests
```

The loop is a generator of events and prints nothing itself, so the interface can be
replaced or switched off without touching the logic.

```bat
python -m unittest discover -s tests
```

## Comparing models

Same harness, same protocol, a different endpoint — so the difference is the model:

```bat
python verify/bench.py --label spark tasks/edit/*.txt tasks/hard/h04_spreadsheet.txt
python verify/bench.py --label big --profile big tasks/edit/*.txt tasks/hard/h04_spreadsheet.txt
python verify/bench.py --label before --rev 3f042c4 tasks/edit/*.txt   :: an older harness
```

Each task runs in a fresh workspace under `~/.miniagent/bench/<label>/` and is scored
by its checker in `verify/` (`SCORE: passed/total`), never by the agent's claim; a
task file starting with `fixture: git <rev>` begins as that revision of this
repository - code to change, with a test suite that has to stay green. Every run is
appended to `~/.miniagent/bench/results.jsonl`, and its trace (steps, tokens, crops,
loops) lands in `~/.miniagent/projects/<slug>/runs/`. An agent's own claim of success is not evidence: during testing
one model wrote a test file with 108 asserts, never called a single test function,
printed `ALL OK` and reported the task complete.

## Licence

MIT — see [LICENSE](LICENSE).
