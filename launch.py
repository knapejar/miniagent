#!/usr/bin/env python3
"""
kaggle-tpu-lab launcher — serve Qwen3.8-27B on a free Kaggle TPU from your terminal.

    python launch.py serve                 # push the kernel and watch it come up
    python launch.py serve --reasoning-effort medium --mtp 3
    python launch.py status                # one-shot status + recent events
    python launch.py shell                 # command channel: over any live tunnel, ntfy as fallback
    python launch.py ssh                   # real SSH into the kernel (cloudflared or pinggy tunnel)
    python launch.py proxy                 # stable http://127.0.0.1:8080/v1 that follows whichever
                                           # tunnel is live (pinggy URLs rotate hourly)
    python launch.py nettest               # test tunnels + channel on a CPU session (no TPU queue)
    python launch.py stop                  # kill the TPU session

ngrok (optional, static URL): set NGROK_URL=https://<name>.ngrok-free.dev and
NGROK_AUTHTOKEN (or have `ngrok config add-authtoken` done on this machine).

Requires the Kaggle CLI, authenticated:  pip install kaggle   (see README).
Only the Python standard library is used here.
"""
import argparse
import http.client
import http.server
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "kernel" / "serve_qwen38.py"
STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"
KEY_DIR = Path.home() / ".kaggle-tpu-lab"            # SSH key + local cloudflared binary
SSH_KEY = KEY_DIR / "id_ed25519"
API_TUNNELS = ("ngrok", "cloudflared", "pinggy")    # tunnels in front of the HTTP API
CMD_WAIT_S = 330   # the kernel lets a shell command run up to 300 s

WEIGHTS_DATASET = "rahim3/qwen3-8-27b-bf16"
ENV_DATASET = "rahim3/qwen38-tpu-env-v5e8"   # XLA compile cache + cloudflared + manifest
NTFY = "https://ntfy.sh"

# Friendly one-liners for each phase the kernel publishes.
PHASE_TEXT = {
    "install":            "Building the Python runtime with uv (~30 s)...",
    "installed":          "Runtime ready.",
    "mtp-patch-applied":  "MTP state-rollback patch applied.",
    "mtp-patch-failed":   "MTP patch did not apply — speculative decoding disabled for safety.",
    "cache-restored":     None,  # rendered below (depends on config coverage)
    "cache-missing":      "No compile cache found — cold compile, add ~10 min.",
    "weights-mounted":    "Weights found mounted (no download needed).",
    "weights-download":   "Downloading weights from Hugging Face (~5 min)...",
    "weights-downloaded": "Weights downloaded.",
    "server-launch":      "Starting vLLM — loading 55 GB of weights, then TPU graph compile...",
    "tunnel-url":         None,
    "tunnel-check":       None,
    "tunnel-restart":     None,
    "tunnel-failed":      None,
    "hold":               None,
    "nettest-result":     None,
    "cmd-ack":            None,
    "cmd-result":         None,
    "compiling":          None,  # rendered with elapsed time below
    "serving":            "Server is HEALTHY.",
    "benchmark":          None,
    "ready":              None,
    "heartbeat":          None,
    "failed":             None,
    "auto-shutdown":      "Keepalive window ended — kernel shut down cleanly.",
    "server-restart":     None,
    "scheduler-fix-failed": "vLLM scheduler fix did not apply — concurrent requests may crash "
                            "the server (it restarts automatically).",
    "stopped":            "Server exited unexpectedly.",
}


def kaggle(*args, capture=True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
    return r


def say(msg):
    print(time.strftime("[%H:%M] "), msg, flush=True)


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit("Kaggle CLI is not working or not authenticated.\n"
                 "Install with `pip install kaggle`, then put your API token in place\n"
                 "(https://www.kaggle.com/settings -> Create New Token).\n\n"
                 f"Error was:\n{(r.stderr or r.stdout).strip()}")


def kaggle_username(cli_arg):
    if cli_arg or os.environ.get("KAGGLE_USERNAME"):
        return cli_arg or os.environ["KAGGLE_USERNAME"]
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    sys.exit("Could not detect your Kaggle username — pass it with --user <name>.")


def ngrok_settings(args):
    """URL from --ngrok-url / NGROK_URL; token from --ngrok-authtoken / NGROK_AUTHTOKEN /
    the local ngrok config file (written by `ngrok config add-authtoken`)."""
    url = args.ngrok_url or os.environ.get("NGROK_URL", "")
    if not url:
        return {}
    tok = args.ngrok_authtoken or os.environ.get("NGROK_AUTHTOKEN", "")
    if not tok:
        home = Path.home()
        for f in (Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"), "ngrok/ngrok.yml"),
                  home / "Library/Application Support/ngrok/ngrok.yml",
                  home / ".config/ngrok/ngrok.yml", home / ".ngrok2/ngrok.yml"):
            m = f.exists() and re.search(r"authtoken:\s*[\"']?([\w-]+)", f.read_text(errors="ignore"))
            if m:
                tok = m.group(1)
                break
    if not tok:
        sys.exit("NGROK_URL is set but no ngrok authtoken was found. Set NGROK_AUTHTOKEN or run\n"
                 "`ngrok config add-authtoken <token>` (dashboard.ngrok.com -> Your Authtoken).")
    return {"ngrok_url": url, "ngrok_authtoken": tok}


def channel_cfg(args):
    """Progress topic + secret command topic + tunnel settings, shared by serve/nettest."""
    cfg = {"ntfy_topic": "ktl-" + uuid.uuid4().hex[:20],
           "ntfy_cmd_topic": "ktl-cmd-" + secrets.token_hex(16),
           "api_key": "sk-" + secrets.token_hex(16),
           **ngrok_settings(args)}
    if args.no_cloudflared:
        cfg["cloudflared"] = False
    if args.no_pinggy:
        cfg["pinggy"] = False
    if not getattr(args, "no_ssh", False):
        cfg["ssh_pubkey"] = ensure_ssh_key()
    return cfg


def ensure_ssh_key():
    """An ed25519 key used only for kernels launched from this machine."""
    pub = SSH_KEY.with_suffix(".pub")
    if not pub.exists():
        if not shutil.which("ssh-keygen"):
            say("WARNING: ssh-keygen not found - the SSH channel is off for this launch.")
            return ""
        KEY_DIR.mkdir(exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "kaggle-tpu-lab",
                        "-f", str(SSH_KEY)], check=True)
    return pub.read_text().strip()


def push_kernel(user, slug, code_file, cfg, tpu=True, datasets=()):
    src = KERNEL_SRC.read_text(encoding="utf-8")
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     lambda _: f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("kernel/serve_qwen38.py is missing the __LAUNCHER_CONFIG__ line")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / code_file).write_text(src, encoding="utf-8")
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true" if tpu else "false",
            "enable_internet": "true",
            "dataset_sources": list(datasets),
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Pushing kernel {user}/{slug} ({'TPU v5e-8' if tpu else 'CPU'})...")
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"WARNING: {line.strip()} — the kernel will still run, "
                    "but may need to download weights / compile cold.")


def save_state(kernel, cfg):
    STATE_FILE.write_text(json.dumps({"kernel": kernel, "topic": cfg["ntfy_topic"],
                                      "cmd_topic": cfg.get("ntfy_cmd_topic", ""),
                                      "api_key": cfg.get("api_key", ""), "tunnels": {}}))


def remember_tunnels(tunnels):
    """Keep the latest URL of every tunnel in the state file, so the command channel,
    the proxy and `ssh` still find the kernel when ntfy stops delivering."""
    urls = {name: (t.get("url") if isinstance(t, dict) else t) for name, t in (tunnels or {}).items()}
    urls = {name: url for name, url in urls.items() if url}
    if not urls or not STATE_FILE.exists():
        return
    try:
        st = json.loads(STATE_FILE.read_text())
    except ValueError:
        return
    if all(st.get("tunnels", {}).get(k) == v for k, v in urls.items()):
        return
    st.setdefault("tunnels", {}).update(urls)
    STATE_FILE.write_text(json.dumps(st))


def channel_hint(cfg):
    say("Backup channel (works without any tunnel): `python launch.py shell`; "
        "stable local endpoint once live: `python launch.py proxy`")
    if cfg.get("ngrok_url"):
        say(f"ngrok URL: {cfg['ngrok_url'].rstrip('/')}/v1  (live once the banner appears)")


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    slug = args.slug
    cfg = {
        **channel_cfg(args),
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "mtp_tokens": args.mtp,
        "reasoning_effort_default": args.reasoning_effort,
        "keepalive_min": args.keepalive_min,
        "debug_hold_min": args.debug_hold_min,
        "weights_dataset": args.weights_dataset,
    }
    if args.no_tools:
        cfg["tool_call_parser"] = ""
    if args.text_only:
        cfg["text_only"] = True
    if args.verbose:
        cfg["verbose"] = True
    if args.fast_start:
        cfg["fast_start"] = True

    push_kernel(user, slug, "serve_qwen38.py", cfg, datasets=[args.weights_dataset, ENV_DATASET])
    save_state(f"{user}/{slug}", cfg)
    say("Pushed. Kaggle takes a few minutes to provision the TPU and attach the "
        "datasets; the endpoint is usually live ~22 min after the kernel starts.")
    channel_hint(cfg)
    say("Watching progress (Ctrl-C is safe — the server keeps running; "
        "`python launch.py status` re-attaches, `... stop` kills it).")
    watch(f"{user}/{slug}", cfg["ntfy_topic"])


def cmd_nettest(args):
    check_auth()
    user = kaggle_username(args.user)
    cfg = {**channel_cfg(args), "net_test": True, "keepalive_min": args.minutes}
    push_kernel(user, args.slug, "nettest.py", cfg, tpu=False)
    save_state(f"{user}/{args.slug}", cfg)
    say("Pushed a CPU kernel: dummy server + tunnels + command channel, no TPU queue.")
    channel_hint(cfg)
    watch(f"{user}/{args.slug}", cfg["ntfy_topic"])


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            events.append((e["time"], json.loads(e.get("message", "{}"))))
        except Exception:
            continue
    return events


def render_event(ev):
    phase = ev.get("phase", "?")
    remember_tunnels(ev.get("tunnels"))
    if phase == "tunnel-restart" and ev.get("name") and ev.get("endpoint"):
        remember_tunnels({ev["name"]: re.sub(r"/v1$", "", ev["endpoint"])})
    if phase == "compiling":
        say(f"Loading / compiling... {ev.get('elapsed_s', 0) // 60} min elapsed "
            "(typically ~20 min with the env dataset, ~35 min without)")
    elif phase == "cache-restored":
        if ev.get("covers_this_config", True):
            say("XLA compile cache restored for this exact config — fast start.")
        else:
            say("XLA compile cache restored, but not for this config — its graphs "
                "compile cold (add ~10 min).")
    elif phase == "tunnel-url":
        say(f"Endpoint URL reserved: {ev.get('endpoint')}  (not live yet — wait for the banner)")
        for name, t in (ev.get("tunnels") or {}).items():
            say(f"   {name:<11} {t.get('url')}  -> {t.get('status')}")
    elif phase in ("tunnel-check", "nettest-result"):
        if phase == "nettest-result":
            say("NETWORK TEST " + ("PASSED" if ev.get("ok") else "FAILED — no tunnel is live")
                + f" (arch {ev.get('arch')}, command channel "
                + ("on" if ev.get("command_channel") else "off") + ", ssh "
                + ("on" if ev.get("ssh") else "off") + ")")
        for name, t in (ev.get("tunnels") or {}).items():
            say(f"   {name:<11} {t.get('url')}  -> {t.get('status')}")
        if phase == "nettest-result":
            say(f"Session stays up {ev.get('keepalive_min')} min: try `python launch.py shell`, "
                "then `python launch.py stop`.")
    elif phase == "tunnel-restart":
        say(f"Tunnel {ev.get('name')} restarted (#{ev.get('restarts')}): "
            f"{ev.get('endpoint')} -> {ev.get('status')}")
    elif phase == "tunnel-failed":
        say(f"No tunnel: {ev.get('note')}")
    elif phase == "hold":
        say(f"Session held {ev.get('minutes')} min for debugging — `python launch.py shell` "
            "(status / log / sh <cmd> / stop).")
    elif phase in ("cmd-ack", "cmd-result"):
        pass
    elif phase == "server-restart":
        say(f"vLLM crashed (rc {ev.get('rc')}) — restarting it in the same TPU session "
            f"({ev.get('restart')}/{ev.get('of')}), ~20 min. `python launch.py cmd log 80` shows why.")
    elif phase == "serving":
        say(f"Server is HEALTHY after {ev.get('startup_secs', 0) // 60} min.")
    elif phase == "benchmark":
        say(f"Quick benchmark: {ev.get('decode_tok_s', '?')} tok/s single-stream decode "
            f"(sanity: {ev.get('sanity', '')!r})")
    elif phase == "ready":
        print("\n" + "=" * 66)
        print("  YOUR ENDPOINT IS LIVE")
        print(f"  base URL : {ev['endpoint']}")
        print(f"  API key  : {ev['api_key']}")
        print(f"  model    : {ev['model']}   (context: {ev.get('max_model_len', '?')})")
        print("=" * 66)
        print("""
Try it:
  curl $BASE/chat/completions -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" -d '{
      "model": "qwen3.8-27b",
      "messages": [{"role": "user", "content": "Hello!"}],
      "chat_template_kwargs": {"reasoning_effort": "low"}
    }'

See the README for hooking this into Claude Code, Codex CLI, opencode, etc.
""")
        say(f"The kernel keeps serving for up to {ev.get('keepalive_min', '?')} min. "
            "Ctrl-C here does NOT stop it; use `python launch.py stop`.")
    elif phase == "heartbeat":
        say(f"Still serving ({ev.get('up_min', '?')} min up) — {ev.get('endpoint', '')}")
    elif phase == "failed":
        say(f"FAILED at step {ev.get('step', '?')}.")
        if ev.get("tail"):
            print("--- last server output ---")
            print(ev["tail"])
        say("Inspect it live: `python launch.py shell` (log / status / sh <cmd>), or the "
            "kernel page on kaggle.com.")
    else:
        text = PHASE_TEXT.get(phase)
        say(text if text else f"{phase} {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")


def watch(kernel, topic):
    since = int(time.time()) - 600
    last_status = None
    seen_boot = False
    try:
        while True:
            for ts, ev in read_events(topic, since):
                since = max(since, ts)
                seen_boot = True
                render_event(ev)
                if ev.get("phase") in ("hold", "auto-shutdown", "stopped"):
                    return
            since = max(since, int(time.time()) - 1) if seen_boot else since
            r = kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            status = m.group(1) if m else "UNKNOWN"
            if status != last_status:
                if status == "QUEUED":
                    say("Kaggle: queued — waiting for a TPU v5e-8 slot...")
                elif status == "RUNNING" and not seen_boot:
                    say("Kaggle: provisioning the VM and attaching datasets "
                        "(a few minutes)...")
                elif status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
                    say(f"Kernel finished with status {status}.")
                    return
                last_status = status
            time.sleep(30)
    except KeyboardInterrupt:
        say("Detached. The kernel keeps running — `python launch.py status` to "
            "re-attach, `python launch.py stop` to kill it.")


def cmd_build_env(args):
    """Maintainer flow. When the kernel finishes:
        kaggle kernels output <user>/<slug> -p bundle_out
        then create/version the dataset from bundle_out/bundle (see README)."""
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-" + uuid.uuid4().hex[:20]
    cfg = {"build_bundle": True, "ntfy_topic": topic, "weights_dataset": args.weights_dataset}
    push_kernel(user, args.slug, "build_env.py", cfg, datasets=[args.weights_dataset])
    save_state(f"{user}/{args.slug}", cfg)
    say(f"Pushed {user}/{args.slug}. It serves each config once (~1.5 h total) and "
        "leaves xla_cache.tar / cloudflared / manifest.json in its output.")
    watch(f"{user}/{args.slug}", topic)


def load_state():
    if not STATE_FILE.exists():
        sys.exit("No launch state found — run `python launch.py serve` first.")
    return json.loads(STATE_FILE.read_text())


def cmd_status(args):
    st = load_state()
    say(f"Kernel: {st['kernel']}")
    r = kaggle("kernels", "status", st["kernel"])
    say(((r.stdout or "") + (r.stderr or "")).strip())
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    for _, ev in events[-8:]:
        render_event(ev)
    if any(ev.get("phase") == "ready" for _, ev in events):
        say(f"API key: {st['api_key']}")
    if args.follow:
        watch(st["kernel"], st["topic"])


def ntfy_stream(topic, since, until, idle_s=15):
    """Yield each JSON payload published on `topic` after unix time `since` exactly once,
    and None when the connection opens or goes idle. A message published while ntfy is
    setting up a live subscription can be missed, so after `idle_s` of silence we
    reconnect and re-read ntfy's cache from `since` (duplicates are dropped)."""
    seen = set()
    while time.time() < until:
        try:
            with urllib.request.urlopen(f"{NTFY}/{topic}/json?since={since}",
                                        timeout=idle_s) as r:
                for raw in r:
                    try:
                        e = json.loads(raw)
                    except ValueError:
                        continue
                    if e.get("event") != "message":
                        yield None
                    elif e["id"] not in seen:
                        seen.add(e["id"])
                        try:
                            yield json.loads(e.get("message") or "{}")
                        except ValueError:
                            pass
                    if time.time() > until:
                        return
        except Exception:
            time.sleep(1)
            yield None


def send_command(st, text, timeout=CMD_WAIT_S):
    if not st.get("cmd_topic"):
        sys.exit("This launch has no command channel (started by an older launch.py).")
    stream = ntfy_stream(st["topic"], int(time.time()) - 2, time.time() + timeout)
    next(stream, None)   # subscribe before publishing, so the reply can't slip past
    req = urllib.request.Request(f"{NTFY}/{st['cmd_topic']}", data=text.encode(), method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        ref = json.load(r)["id"]
    t_sent, acked, parts = time.time(), False, {}
    for ev in stream:
        if ev is None or ev.get("ref") != ref:
            if not acked and time.time() - t_sent > 60:
                break
            continue
        if ev.get("phase") == "cmd-ack":
            acked = True
        elif ev.get("phase") == "cmd-result":
            parts[ev["part"]] = ev["out"]
            if len(parts) == ev["parts"]:
                return ev.get("rc"), "".join(parts[i] for i in range(ev["parts"])).rstrip()
    return None, ("(no answer — the kernel is not running yet, or has ended)" if not acked
                  else "(timed out waiting for the output)")


def http_json(url, token, data=None, timeout=30):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Authorization": f"Bearer {token}",
                                          "User-Agent": "kaggle-tpu-lab",
                                          "ngrok-skip-browser-warning": "1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def api_urls(st):
    tunnels = st.get("tunnels") or {}
    return [tunnels[n] for n in API_TUNNELS if tunnels.get(n)]


def fetch_tunnels(st):
    """{name: {url, status}} from the kernel's front server over a known tunnel, else
    from an ntfy `tunnel` command. Refreshes the state file."""
    for base in api_urls(st):
        try:
            tunnels = http_json(base + "/ktl/tunnels", st["api_key"], timeout=20)["tunnels"]
            remember_tunnels(tunnels)
            st["tunnels"] = {**st.get("tunnels", {}),
                             **{n: t["url"] for n, t in tunnels.items() if t.get("url")}}
            return tunnels
        except Exception:
            continue
    rc, out = send_command(st, "tunnel", timeout=90)
    tunnels = {}
    for line in (out or "").splitlines():
        m = re.match(r"([\w-]+): (\S+) -> (.*)", line.strip())
        if m and m.group(2) != "None":
            tunnels[m.group(1)] = {"url": m.group(2), "status": m.group(3)}
    remember_tunnels(tunnels)
    return tunnels


def run_remote(st, text, timeout=CMD_WAIT_S):
    """Run a control command: over a live tunnel first, ntfy as the fallback."""
    for base in api_urls(st):
        try:
            j = http_json(base + "/ktl/cmd", st["api_key"], text.encode(), timeout=timeout)
            return j.get("rc"), j.get("out", "")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return 1, "(the kernel rejected the key - is this the right launch?)"
        except Exception:
            continue
    return send_command(st, text, timeout)


def cmd_cmd(args):
    rc, out = run_remote(load_state(), " ".join(args.text))
    print(out)
    sys.exit(1 if rc is None else rc)


def cmd_shell(args):
    st = load_state()
    say(f"Command channel to {st['kernel']} via ntfy — `help` lists commands, "
        "`exit` or Ctrl-C quits (the kernel keeps running).")
    try:
        while True:
            line = input("kaggle> ").strip()
            if line in ("exit", "quit"):
                break
            if line:
                rc, out = run_remote(st, line)
                print(out)
                if rc not in (0, None):
                    print(f"[exit {rc}]")
    except (KeyboardInterrupt, EOFError):
        print()


TUNNEL_PREF = ("ngrok", "cloudflared", "pinggy")
# Qwen3.8's chat template knows xhigh | medium | low; Claude Code sends low | medium |
# high | xhigh | max in output_config.effort (default high) and vLLM rejects the rest.
EFFORT_MAP = {"high": "medium", "max": "xhigh", "minimal": "low", "none": "low"}


def adapt_body(path, body):
    """Make requests from Anthropic-API clients acceptable to vLLM + Qwen3.8."""
    if not body or not path.startswith("/v1/messages"):
        return body
    try:
        data = json.loads(body)
    except ValueError:
        return body
    effort = (data.get("output_config") or {}).get("effort")
    if effort in EFFORT_MAP:
        data["output_config"]["effort"] = EFFORT_MAP[effort]
        return json.dumps(data).encode()
    return body


def live_tunnels(st):
    """The kernel's public API URLs that answer right now, best first."""
    tunnels = fetch_tunnels(st)
    found = {n: t["url"] for n, t in tunnels.items() if n in TUNNEL_PREF and t.get("url")
             and (t.get("status") == "live" or str(t.get("status")).startswith("tunnel-ok"))}
    return [found[n] for n in TUNNEL_PREF if n in found]


def cmd_proxy(args):
    """Local reverse proxy: clients talk to http://127.0.0.1:<port>/v1 with any API key;
    requests go to whichever tunnel is live, re-resolved over ntfy when one dies."""
    st = load_state()
    cur = {"urls": [], "at": 0}
    lock = __import__("threading").Lock()

    def resolve(force=False):
        with lock:
            if force or not cur["urls"]:
                if time.time() - cur["at"] < 10 and cur["urls"]:
                    return cur["urls"]
                say("Resolving the live tunnel over the command channel...")
                cur["urls"], cur["at"] = live_tunnels(st), time.time()
                say(f"Upstream: {cur['urls'][0] if cur['urls'] else 'NONE LIVE (retrying on next request)'}")
            return cur["urls"]

    hop = {"host", "connection", "keep-alive", "transfer-encoding", "te", "trailer",
           "upgrade", "proxy-authorization", "proxy-connection", "content-length",
           "authorization", "accept-encoding"}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"     # the connection close delimits streamed bodies

        def log_message(self, fmt, *a):
            say(f"proxy {self.command} {self.path} " + (fmt % a if "%" in fmt else ""))

        def forward(self):
            body = adapt_body(self.path, self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in hop}
            headers["Authorization"] = f"Bearer {st['api_key']}"
            headers["ngrok-skip-browser-warning"] = "1"
            last = "no live tunnel"
            for attempt in range(4):
                urls = resolve(force=attempt > 0)
                if not urls:
                    time.sleep(15)
                    continue
                u = urllib.parse.urlsplit(urls[0])
                conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=args.timeout)
                try:
                    conn.request(self.command, self.path, body=body or None,
                                 headers={**headers, "Host": u.hostname})
                    r = conn.getresponse()
                except Exception as e:
                    last = f"{urls[0]}: {e}"
                    conn.close()
                    cur["urls"] = []
                    time.sleep(5)
                    continue
                if r.status in (502, 503, 504, 530) or (
                        r.status == 404 and r.getheader("Content-Type", "").startswith("text/html")):
                    last = f"{urls[0]}: tunnel edge HTTP {r.status}"
                    conn.close()
                    cur["urls"] = []
                    time.sleep(5)
                    continue
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in hop:
                        self.send_header(k, v)
                self.end_headers()
                try:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    conn.close()
                return
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"kaggle-tpu-lab proxy: {last}"}).encode())

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = forward

    resolve()
    srv = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print("\n" + "=" * 66)
    print(f"  LOCAL PROXY   base URL : http://{args.host}:{args.port}/v1")
    print("                API key  : anything (the proxy adds the real one)")
    print("                model    : qwen3.8-27b")
    print(f"  Claude Code:  ANTHROPIC_BASE_URL=http://{args.host}:{args.port}")
    print("=" * 66 + "\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        say("Proxy stopped (the kernel keeps running).")


def local_cloudflared():
    """cloudflared on this machine, for `cloudflared access ssh` (plain HTTPS, port 443)."""
    found = shutil.which("cloudflared")
    if found:
        return found
    name = {"win32": "cloudflared-windows-amd64.exe", "linux": "cloudflared-linux-amd64"}.get(
        sys.platform)
    if not name:
        sys.exit("Install cloudflared (macOS: brew install cloudflared), or use --via pinggy.")
    dest = KEY_DIR / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    if not dest.exists():
        KEY_DIR.mkdir(exist_ok=True)
        say(f"Downloading {name} (one time)...")
        urllib.request.urlretrieve(
            f"https://github.com/cloudflare/cloudflared/releases/latest/download/{name}", dest)
        dest.chmod(0o755)
    return str(dest)


def ssh_command(tunnels, via=None):
    """The ssh argv for the first usable SSH tunnel, or None."""
    if not SSH_KEY.exists():
        sys.exit(f"No SSH key at {SSH_KEY} - it is created by `serve`/`nettest` launches.")
    base = ["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
            # kernels get fresh host keys every launch; a real file (not os.devnull, which
            # Git Bash's ssh turns into a file named "nul") keeps ssh quiet
            "-o", f"UserKnownHostsFile={(KEY_DIR / 'known_hosts').as_posix()}",
            "-o", "ServerAliveInterval=30",
            "-o", "LogLevel=ERROR"]
    for name in ([f"{via}-ssh"] if via else ["cloudflared-ssh", "pinggy-ssh"]):
        url = (tunnels.get(name) or {}).get("url")
        if not url:
            continue
        if name == "pinggy-ssh":
            m = re.match(r"tcp://([^:/]+):(\d+)", url)
            if m:
                return name, base + ["-p", m.group(2), f"root@{m.group(1)}"]
        else:
            host = re.sub(r"^https?://", "", url).rstrip("/")
            # forward slashes: both Windows OpenSSH and Git Bash's ssh (which runs the
            # proxy command through sh and eats backslashes) accept them
            cf = Path(local_cloudflared()).as_posix()
            cf = f'"{cf}"' if " " in cf else cf
            return name, base + ["-o", f"ProxyCommand={cf} access ssh --hostname %h",
                                 f"root@{host}"]
    return None, None


def cmd_ssh(args):
    st = load_state()
    tunnels = fetch_tunnels(st)
    name, argv = ssh_command(tunnels, args.via)
    if not argv:
        sys.exit("No SSH tunnel is known for this launch (was it started with an SSH key? "
                 "`python launch.py cmd tunnel` lists the tunnels).")
    say(f"SSH via {name}: {tunnels[name]['url']}")
    sys.exit(subprocess.call(argv + list(args.remote)))


def cmd_stop(args):
    st = load_state()
    # graceful stop over a tunnel or ntfy first; deleting the kernel is the fallback
    for base in api_urls(st):
        try:
            http_json(base + "/ktl/cmd", st["api_key"], b"stop", timeout=20)
            break
        except Exception:
            continue
    if st.get("cmd_topic"):
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{NTFY}/{st['cmd_topic']}", data=b"stop", method="POST"), timeout=15)
        except Exception:
            pass
    say(f"Deleting kernel {st['kernel']} (terminates the TPU session)...")
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "delete",
                        st["kernel"]], input="yes\n", capture_output=True, text=True)
    say((p.stdout + p.stderr).strip() or "done")


def add_tunnel_args(s):
    s.add_argument("--ngrok-url", help="static ngrok URL, e.g. https://name.ngrok-free.dev "
                   "(default: env NGROK_URL)")
    s.add_argument("--ngrok-authtoken", help="default: env NGROK_AUTHTOKEN or the local ngrok config")
    s.add_argument("--no-cloudflared", action="store_true",
                   help="don't open the backup trycloudflare tunnel")
    s.add_argument("--no-pinggy", action="store_true",
                   help="don't open the pinggy SSH tunnel over port 443 (no-account fallback)")
    s.add_argument("--no-ssh", action="store_true",
                   help="don't start sshd and its tunnels in the kernel")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="push the serving kernel and watch it come up")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-tpu-serve", help="kernel name")
    s.add_argument("--max-model-len", type=int, default=262144,
                   help="context length (default: native 262k; use 131072 with "
                        "--max-num-seqs 16 for max multi-stream throughput)")
    s.add_argument("--max-num-seqs", type=int, default=4)
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (0 disables). +34%% decode in our A/B test; made "
                        "lossless by the bundled GDN state-rollback patch "
                        "(verified 12/12 greedy exact-match)")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "medium", "low"],
                   help="server-side default; clients can still override per request")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower: ~8 min faster start, image inputs "
                        "then error out")
    s.add_argument("--verbose", action="store_true",
                   help="show every vLLM log line in the kernel log")
    s.add_argument("--debug-hold-min", type=int, default=30,
                   help="on failure keep the session alive this long for `launch.py shell`")
    add_tunnel_args(s)
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile: endpoint live in ~4 min (with the env "
                        "dataset), common request shapes are warmed right after; an "
                        "unusual request shape stalls ~1 min the first time")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("nettest", help="CPU kernel (no TPU queue) that tests the tunnels and "
                       "the command channel against a dummy HTTP server")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="ktl-nettest")
    s.add_argument("--minutes", type=int, default=20, help="how long the test session stays up")
    add_tunnel_args(s)
    s.set_defaults(fn=cmd_nettest)

    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                       "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.set_defaults(fn=cmd_build_env)

    s = sub.add_parser("status", help="show current kernel status + recent events")
    s.add_argument("--follow", "-f", action="store_true", help="keep watching")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("shell", help="interactive command channel to the running kernel (ntfy)")
    s.set_defaults(fn=cmd_shell)

    s = sub.add_parser("cmd", help="run one command on the kernel: status | log [N] | "
                       "tunnel [restart] | sh <cmd> | stop")
    s.add_argument("text", nargs="+")
    s.set_defaults(fn=cmd_cmd)

    s = sub.add_parser("ssh", help="SSH into the running kernel as root (over a tunnel)")
    s.add_argument("--via", choices=["cloudflared", "pinggy"],
                   help="which SSH tunnel to use (default: cloudflared, then pinggy)")
    s.add_argument("remote", nargs=argparse.REMAINDER, help="optional command to run remotely")
    s.set_defaults(fn=cmd_ssh)

    s = sub.add_parser("proxy", help="local stable endpoint that forwards to the live tunnel")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--timeout", type=int, default=1800, help="upstream timeout per request (s)")
    s.set_defaults(fn=cmd_proxy)

    s = sub.add_parser("stop", help="terminate the TPU session")
    s.set_defaults(fn=cmd_stop)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
