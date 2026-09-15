# kaggle-tpu-lab

Run open models on Kaggle's free TPU v5e-8 and get a public endpoint that speaks the
OpenAI and Anthropic APIs. Point Claude Code, Codex CLI, opencode or anything else at it.
No GPU, no cloud bill, about twenty minutes from pressing Run to a URL.

Each model has its own folder with a run-all Kaggle notebook, the kernel script behind
it, and a write-up of how it works and what we measured.

| Model | Weights on the TPU | Context | One stream | Many streams | Prefill | Run → URL | Engine | |
|---|---|---|---|---|---|---|---|---|
| [Qwen3.8-27B](qwen38-27b/) | bf16, no quantization | 262k | ~130 tok/s | ~540 tok/s at 8 | 10,300 tok/s | ~22 min | vllm-tpu + one patch | [notebook](https://www.kaggle.com/code/rahim3/qwen3-8-27b-bf16-on-kaggle-tpu-130-tok-s-api) |
| [GLM-5.3-Flash](glm53-flash/) (320B MoE) | 3-bit experts, int8 rest | 262k | ~64 tok/s | ~90 tok/s at 3 | ~1,600 tok/s | ~16 min | our own JAX engine | [notebook](https://www.kaggle.com/code/rahim3/glm-5-3-flash-on-a-free-kaggle-tpu-64-tok-s-api) |

Numbers are measured on the shipped configuration; the folder READMEs say how. Qwen runs on
vllm-tpu with one patch. GLM-5.3-Flash runs on an engine we wrote in JAX for it; as far as we
know it is the first to run that model on a TPU.

## What you need

A Kaggle account with TPU access (phone-verify it under Settings) and its free quota,
around 20 TPU hours a week. Nothing to install for the notebook route. For the terminal
route, Python 3.9+ and the Kaggle CLI.

## How a session works

The notebook attaches public datasets holding the weights (and, where it helps, a
pre-built compile cache), builds the engine across the eight chips, opens a Cloudflare
tunnel and prints the URL and an API key. The last cell is the server: leave it running.
A keepalive holds the session up to Kaggle's limit, a little under nine hours, after
which you run it again and get a new URL.

Wire a coding agent with one line, for example Claude Code:

```bash
ANTHROPIC_BASE_URL=<url> ANTHROPIC_AUTH_TOKEN=<key> ANTHROPIC_MODEL=<model> claude
```

Each folder README has the exact lines for Claude Code, Codex CLI and opencode.

## From a terminal

```bash
git clone https://github.com/knapejar/miniagent      # this lab lives in miniagent/kaggle-tpu-lab
cd miniagent/kaggle-tpu-lab
python launch.py serve                               # Qwen3.8-27B
python launch.py serve --model glm53-flash           # GLM-5.3-Flash
```

`launch.py` pushes the kernel with the Kaggle CLI and follows its progress; `status`
and `stop` do what they say, `proxy`, `cmd` and `ssh` reach the running kernel (below).
The same kernels also run under DeployMan (`services/qwen38-api`, `services/glm53-api`),
built with `python tools/build_kernel.py <model> <file>`.

## Control channels, tunnels and SSH

Every launch opens its tunnels in the first seconds of the kernel (before the runtime
installs), and every tunnel points at a small **front server** inside the kernel, not at
the model server directly. The front server forwards the OpenAI/Anthropic API to it and answers its
own control routes, so **a live tunnel is always also a command channel** — while the model
loads or compiles, after it crashed, or when a later step failed.

| Tunnel | Needs | Port | Carries |
|---|---|---|---|
| **ngrok** | free account + static domain | 443 | API + control, URL never changes |
| **cloudflared** quick tunnel | nothing | **7844** (blocked on some networks) | API + control |
| **pinggy** (SSH reverse tunnel) | nothing | 443 | API + control, URL rotates every 60 min |
| **cloudflared-ssh** | nothing | 7844 on the kernel side, 443 on yours | SSH to the kernel |
| **pinggy-ssh** (TCP) | nothing | 443 | SSH to the kernel |

```bash
python launch.py proxy       # http://127.0.0.1:8080/v1, any API key; follows whichever tunnel is live
python launch.py shell       # status | log 100 | tunnel | sh <cmd> | stop
python launch.py cmd sh "free -g; ps aux | grep vllm"
python launch.py ssh         # root shell in the kernel (cloudflared first, then pinggy)
python launch.py ssh --via pinggy nvidia-smi   # any remote command
```

How the launcher reaches the kernel, in order:

1. **HTTPS to a known tunnel** (`/ktl/cmd`, `/ktl/tunnels`, authenticated with the launch's
   API key). The launcher stores every tunnel URL it has seen in `~/.kaggle-tpu-lab.json`.
2. **ntfy.sh** as the fallback (commands on a secret topic, replies on the progress topic).
   ntfy.sh is a free shared service and rate-limits busy IPs — on 2026-09-15 it stopped
   delivering to a TPU kernel mid-session, which is why it is no longer the primary channel.
3. **SSH**: `serve` and `nettest` create `~/.kaggle-tpu-lab/id_ed25519` once and put its
   public key into the kernel config; the kernel installs `openssh-server` if needed and runs
   `sshd` on `127.0.0.1:2222` behind its own tunnels. `launch.py ssh` downloads a local
   `cloudflared` once for `cloudflared access ssh` (plain HTTPS). `--no-ssh` turns it off.

Front-server routes (all behind any API tunnel):

| Route | Auth | Answer |
|---|---|---|
| `GET /ktl/health` | none | `{"ktl": true, "server": <vLLM up>, "phase": ..., "uptime": ...}` |
| `GET /ktl/tunnels` | `Authorization: Bearer <api key>` | every tunnel with its URL and status |
| `POST /ktl/cmd` | same | body = command (`status`, `log 80`, `tunnel`, `sh <cmd>`, `stop`) → `{"rc", "out"}` |
| anything else | the model server checks the key | forwarded to the model server on `127.0.0.1:8001`, streaming |

For a URL that never changes, claim the free static domain in the ngrok dashboard:

```bash
export NGROK_URL="https://<name>.ngrok-free.dev"     # Windows: setx NGROK_URL https://...
export NGROK_AUTHTOKEN="<token>"
python launch.py nettest     # ~5 min CPU session, no TPU queue: checks every tunnel, the channels and SSH
python launch.py serve       # ngrok is the primary endpoint, the others stay as backups
python launch.py cmd tunnel ngrok https://<name>.ngrok-free.dev <authtoken>   # or add it to a running session
```

Both kernels hold the session for `--debug-hold-min` (default 30) when a step fails, so you
can look around with `shell` or `ssh` instead of losing the TPU slot; Qwen also restarts a
crashed vLLM in place. The Qwen kernel carries this code inline (verified on Kaggle); the GLM
kernel gets the same code from [`common/ktl_control.py`](common/ktl_control.py), embedded by the
launcher. See [docs/operations.md](docs/operations.md) for
the recovery runbook and the incidents behind these changes.


## Adding a model

One folder at the top level, named after the model: `README.md` with the numbers and the
how, `kernel/` with the serving script, `notebook/` with the run-all notebook generated
from it, plus whatever the recipe needs (a patch, an engine). The launcher and the
notebook share the same config block, so a setting changed in one place means the same
thing in the other. Load `common/ktl_control.py` before anything slow (see
`glm53-flash/kernel/serve_glm53.py`, `load_control`) and add the model to `MODELS` in `launch.py`.

```
launch.py                 the CLI: serve [--model] / status / shell / cmd / ssh / proxy / nettest / stop
common/ktl_control.py     front server, tunnels, SSH, /ktl/cmd, debug hold (embedded into kernels)
docs/operations.md        control channels, SSH, recovery runbook, incident notes
tools/build_kernel.py     a self-contained kernel script for other runners (DeployMan)
tools/pack_notebook.py    a model's notebook from its kernel
qwen38-27b/               Qwen3.8-27B on vllm-tpu
glm53-flash/              GLM-5.3-Flash on our JAX engine
```

## License

The code here is MIT. Model weights keep their own licenses; each folder says which.
