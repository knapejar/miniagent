"""
Control plane for a kaggle-tpu-lab kernel, independent of the model server.

The Qwen kernel carries the same code inline (qwen38-27b/kernel/serve_qwen38.py, verified on Kaggle);
this module is that code made reusable, and launch.py embeds it into kernels that import it
(glm53-flash). It gives a kernel, from its first seconds:

  * a front server on 127.0.0.1:PORT that every tunnel points at: /ktl/health, /ktl/tunnels and
    /ktl/cmd (status, log, tunnel, sh, stop) answered here, everything else forwarded to the model
    server on BACKEND_PORT, streaming; while that server is down the API answers 503 JSON;
  * public tunnels: ngrok (static domain, if configured), a cloudflared quick tunnel, pinggy over
    port 443, restarted by a watchdog;
  * SSH: sshd on 127.0.0.1:2222 for one public key, behind cloudflared-ssh and pinggy-ssh;
  * ntfy.sh commands as the fallback channel, and a debug hold on failure.

Use:
    import ktl_control as ktl
    ktl.setup(CFG, log=log, raw_log=path, backend_port=8001, t0=T0)
    ktl.publish(phase, **extra)          # progress event: log + ntfy + the phase /ktl/health reports
    ktl.start()                          # front server + tunnels/SSH in the background
    ...
    ktl.primary_url(), ktl.tunnel_summary(), ktl.hold_for_debug(), ktl.shutdown(rc)
"""
import collections
import http.client
import http.server
import io
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULTS = {
    "api_key": "",
    "ntfy_topic": "",              # optional: publish progress to ntfy.sh/<topic>
    "ntfy_cmd_topic": "",          # optional: accept commands from ntfy.sh/<topic> (keep it secret)
    "debug_hold_min": 30,          # on failure keep the session alive this long (needs a tunnel, SSH or ntfy)
    "ngrok_url": "",               # e.g. https://name.ngrok-free.dev (your free static domain)
    "ngrok_authtoken": "",         # or env / Kaggle secret NGROK_AUTHTOKEN
    "ngrok_pooling": True,         # --pooling-enabled: a stale agent can't block the URL
    "cloudflared": True,           # a trycloudflare quick tunnel
    "cloudflared_protocol": "http2",  # TCP 7844 (quic = UDP 7844); either port may be blocked
    "pinggy": True,                # no-account fallback over 443 (free tunnels expire after 60 min)
    "ssh_pubkey": "",              # authorized key for root; enables sshd behind its own tunnels
    "net_test": False,             # CPU smoke test: dummy backend + tunnels + channels, no model
}

CFG = dict(DEFAULTS)
PORT = 8000                        # front server: what every tunnel points at
BACKEND_PORT = 8001                # the model server, reached through the front server
SSH_PORT = 2222
CMD_TIMEOUT = 300                  # seconds a `sh` command from the channel may run
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
CLOUDFLARED = Path("/tmp/cloudflared")
NGROK = Path("/tmp/ngrok")
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64",
        "arm64": "arm64"}.get(platform.machine().lower(), "amd64")
STATE = {"phase": "boot"}
TUNNELS = {}   # name -> {url, state, note, probe, proc, cmd, env, parse, restarts}
HOOKS = {"server_status": None, "on_stop": None}
T0 = time.time()
RAW_LOG = WORK / "server.log"
_raw = None
_log = None
_lock = threading.Lock()


def setup(cfg, log=None, raw_log=None, port=PORT, backend_port=BACKEND_PORT, t0=None,
          server_status=None, on_stop=None):
    """cfg: the kernel's config dict (control keys missing from it get DEFAULTS, written back).
    log: the kernel's log(*parts). raw_log: file every log line and tool output goes to (`log N`).
    server_status(): one line about the model server for `status`. on_stop(): before shutdown."""
    global PORT, BACKEND_PORT, T0, RAW_LOG, _raw, _log, CFG
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    CFG = cfg
    PORT, BACKEND_PORT = int(port), int(backend_port)
    T0 = t0 or time.time()
    RAW_LOG = Path(raw_log) if raw_log else RAW_LOG
    _raw = open(RAW_LOG, "a", buffering=1)
    _log = log
    HOOKS.update(server_status=server_status, on_stop=on_stop)


def raw(line):
    try:
        with _lock:
            _raw.write(line if line.endswith("\n") else line + "\n")
    except Exception:  # noqa: BLE001
        pass


def log(*parts):
    if _log:
        _log(*parts)
    else:
        line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
        print(line, flush=True)
        raw(line)


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def ntfy_send(payload):
    """Push one JSON event to the progress topic (ntfy.sh caps a message at 4 KB)."""
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {payload['phase']}",
                "message": json.dumps(payload)}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # noqa: BLE001
        log(f"(ntfy publish failed: {e})")


def publish(phase, **extra):
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    STATE["phase"] = phase
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    ntfy_send({"phase": phase, **extra})


def log_tail(n):
    try:
        with open(RAW_LOG, errors="ignore") as f:
            return "".join(collections.deque(f, maxlen=n))
    except OSError as e:
        return f"(cannot read {RAW_LOG}: {e})"


def hold_for_debug():
    """Keep the Kaggle session (and every channel) alive for a post-mortem."""
    channels = bool(CFG["ntfy_cmd_topic"] or CFG["ssh_pubkey"] or TUNNELS)
    mins = CFG["debug_hold_min"] if channels else 0
    if mins > 0:
        publish("hold", minutes=mins)
        log(f"   holding the session {mins} min for debugging — launch.py cmd / ssh over a tunnel, "
            "or ntfy (status / log / sh <cmd> / stop)")
        time.sleep(mins * 60)


def shutdown(rc):
    """Stop the model server hook and every tunnel, then exit from any thread."""
    try:
        if HOOKS["on_stop"]:
            HOOKS["on_stop"]()
    except Exception:  # noqa: BLE001
        pass
    for t in list(TUNNELS.values()):
        if t["proc"].poll() is None:
            t["proc"].terminate()
    time.sleep(3)
    os._exit(rc)


def install_binary(dest, fetch):
    """Write to a private temp file, then rename: a half-written binary never ends up at `dest`."""
    part = Path(f"{dest}.part-{threading.get_ident()}")
    try:
        fetch(part)
        part.chmod(0o755)
        os.replace(part, dest)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"({dest.name} install failed: {e})")
        part.unlink(missing_ok=True)
        return False


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return True
    return install_binary(CLOUDFLARED, lambda p: urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{ARCH}", p))


def fetch_ngrok():
    if NGROK.exists():
        return True

    def get(part):
        url = f"https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-{ARCH}.tgz"
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            part.write_bytes(tf.extractfile("ngrok").read())
    return install_binary(NGROK, get)


# ---------------- tunnels ----------------
def ngrok_authtoken():
    tok = CFG["ngrok_authtoken"] or os.environ.get("NGROK_AUTHTOKEN", "")
    if tok or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return tok
    box = []

    def from_kaggle_secrets():   # notebook flow: Add-ons -> Secrets -> NGROK_AUTHTOKEN
        try:
            from kaggle_secrets import UserSecretsClient
            box.append(UserSecretsClient().get_secret("NGROK_AUTHTOKEN"))
        except Exception:  # noqa: BLE001
            pass
    th = threading.Thread(target=from_kaggle_secrets, daemon=True)
    th.start()
    th.join(20)
    return box[0] if box else ""


def _spawn(t):
    t["state"], t["note"], t["probe"] = "starting", "", ""
    t["proc"] = subprocess.Popen(t["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, text=True, env=t["env"])

    def pump(proc=t["proc"]):
        for line in proc.stdout:
            raw(f"[{t['name']}] {line}")
            t["parse"](t, line.strip())
    threading.Thread(target=pump, daemon=True).start()
    TUNNELS[t["name"]] = t
    return t


def start_ngrok():
    url = CFG["ngrok_url"].strip().rstrip("/")
    if not url:
        return None
    if not url.startswith("http"):
        url = "https://" + url
    tok = ngrok_authtoken()
    if not tok:
        log("   ngrok_url is set but there is no authtoken (config ngrok_authtoken, env or "
            "Kaggle secret NGROK_AUTHTOKEN) -> skipping ngrok")
        return None
    if not fetch_ngrok():
        return None
    cmd = [str(NGROK), "http", str(PORT), "--url", url, "--log", "stdout", "--log-format", "json"]
    if CFG["ngrok_pooling"]:
        cmd.append("--pooling-enabled")

    def parse(t, line):
        try:
            j = json.loads(line)
        except ValueError:
            j = {}
        if j.get("msg") == "started tunnel":
            t["state"], t["note"] = "online", ""
        elif "ERR_NGROK" in line or j.get("lvl") in ("eror", "crit"):
            t["state"] = "error"
            t["note"] = str(j.get("err") or j.get("msg") or line)[:300]
    # token via env, not argv, so it doesn't show up in `ps`
    return _spawn({"name": "ngrok", "url": url, "cmd": cmd, "parse": parse, "restarts": 0,
                   "env": {**os.environ, "NGROK_AUTHTOKEN": tok}})


def port_open(port, host="127.0.0.1", timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_cloudflared(name="cloudflared", target=None):
    if not CFG["cloudflared"] or not fetch_cloudflared():
        return None
    target = target or f"http://127.0.0.1:{PORT}"
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

    def parse(t, line):
        m = pat.search(line)
        if m and not t["url"]:
            t["url"] = m.group(0)
        if "Registered tunnel connection" in line:
            t["state"], t["note"] = "online", ""
        elif " ERR " in line and t["state"] != "online":
            t["note"] = line[-300:]
    return _spawn({"name": name, "url": None, "parse": parse, "restarts": 0, "env": None,
                   "cmd": [str(CLOUDFLARED), "tunnel", "--url", target,
                           "--no-autoupdate", "--protocol", CFG["cloudflared_protocol"]]})


def apt_install(package, binary, budget_s=480):
    """apt-get install, waiting for the dpkg lock: another installer (a DeployMan bootstrap) may hold it."""
    found = shutil.which(binary) or (f"/usr/sbin/{binary}" if Path(f"/usr/sbin/{binary}").exists() else None)
    if found:
        return found
    log(f"   {binary} missing -> apt-get install {package}")
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    lock = ["-o", "DPkg::Lock::Timeout=180"]
    t_end = time.time() + budget_s
    while time.time() < t_end:
        for cmd in (["apt-get", "install", "-y", "-qq", "--no-install-recommends", *lock, package],
                    ["apt-get", "update", "-qq", *lock],
                    ["apt-get", "install", "-y", "-qq", "--no-install-recommends", *lock, package]):
            try:
                if subprocess.run(cmd, capture_output=True, timeout=400, env=env).returncode == 0 \
                        and cmd[1] == "install":
                    break
            except Exception as e:  # noqa: BLE001
                log(f"   ({' '.join(cmd[:2])} failed: {e})")
        found = shutil.which(binary) or (f"/usr/sbin/{binary}" if Path(f"/usr/sbin/{binary}").exists() else None)
        if found:
            return found
        time.sleep(15)
    return None


def start_pinggy(name="pinggy", port=None, tcp=False):
    """Reverse SSH tunnel over port 443: needs no account and no port but 443, so it survives
    networks that block cloudflared's 7844. Free tunnels last 60 min; the watchdog reconnects."""
    if not CFG["pinggy"]:
        return None
    ssh = apt_install("openssh-client", "ssh", budget_s=240)
    if not ssh:
        log("   no ssh client -> skipping pinggy")
        return None
    if tcp:
        pat = re.compile(r"tcp://[a-z0-9.-]+:\d+")
    else:
        pat = re.compile(r"https://[a-z0-9-]+\.(?:[a-z0-9-]+\.)*(?:pinggy\.net|pinggy-free\.link|pinggy\.link)")

    def parse(t, line):
        m = pat.search(line)
        if m and not t["url"] and "dashboard." not in m.group(0):
            t["url"], t["state"], t["note"] = m.group(0), "online", ""
        elif "expire" in line or "denied" in line.lower() or "error" in line.lower():
            t["note"] = line[-200:]
    return _spawn({"name": name, "url": None, "parse": parse, "restarts": 0, "env": None,
                   "cmd": [ssh, "-p", "443", "-T", "-o", "StrictHostKeyChecking=no",
                           "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=30",
                           "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes",
                           "-o", "ConnectTimeout=20",
                           "-R", f"0:127.0.0.1:{port or PORT}",
                           ("tcp@a.pinggy.io" if tcp else "a.pinggy.io")]})


def http_get(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle-tpu-lab", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(600).decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read(600).decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:200]


def probe_tunnel(t):
    """Request the public URL from inside the kernel: tells a working tunnel in front of a
    not-yet-started server apart from a broken tunnel."""
    if t["proc"].poll() is not None:
        v = f"exited rc={t['proc'].returncode} {t['note']}".strip()
    elif not t["url"]:
        v = f"no URL yet {t['note']}".strip()
    elif t["name"].endswith("-ssh"):
        v = t["state"]                      # sshd behind it; nothing to GET
    else:
        code, body = http_get(t["url"] + "/ktl/health", {"ngrok-skip-browser-warning": "1"})
        try:
            health = json.loads(body) if code == 200 else {}
        except ValueError:
            health = {}
        if health.get("ktl"):
            v = "live" if health.get("server") else "tunnel-ok (server not up yet)"
        else:
            v = f"broken (HTTP {code}: {' '.join(body.split())[:160]})"
    t["probe"] = v
    return v


def primary_url():
    rank = lambda t: 0 if t["probe"] == "live" else 1 if t["probe"].startswith("tunnel-ok") else 2  # noqa: E731
    ok = sorted((t for t in list(TUNNELS.values()) if t["url"] and not t["name"].endswith("-ssh")),
                key=rank)   # stable: ngrok first
    return ok[0]["url"] if ok else None


def tunnel_summary():
    return {n: {"url": t["url"], "status": t["probe"] or t["state"]} for n, t in list(TUNNELS.items())}


def probe_all():
    for t in list(TUNNELS.values()):
        probe_tunnel(t)
    return tunnel_summary()


WATCHDOG = []
SSH_BOOT = []
BOOT = {}


def start_watchdog():
    if not WATCHDOG:
        WATCHDOG.append(threading.Thread(target=tunnel_watchdog, daemon=True))
        WATCHDOG[0].start()


def tunnel_watchdog():
    while True:
        time.sleep(20)
        for t in list(TUNNELS.values()):
            if t["proc"].poll() is None or TUNNELS.get(t["name"]) is not t:
                continue          # alive, or replaced by a `tunnel <name>` command
            t["restarts"] += 1
            log(f"   {t['name']} exited (rc={t['proc'].returncode}) {t['note']} -> restarting")
            # pinggy's free tunnel ends every 60 min by design: reconnect right away
            time.sleep(3 if t["name"].startswith("pinggy") else min(300, 10 * t["restarts"]))
            if t["name"].startswith(("cloudflared", "pinggy")):
                t["url"] = None           # these get a new URL on every start
            _spawn(t)
            for _ in range(60):
                if t["url"] and t["state"] != "starting":
                    break
                time.sleep(2)
            probe_tunnel(t)
            publish("tunnel-restart", name=t["name"], restarts=t["restarts"],
                    endpoint=(t["url"] if t["name"].endswith("-ssh") else f"{t['url']}/v1")
                    if t["url"] else None, status=t["probe"], tunnels=tunnel_summary())


def start_sshd():
    """Root sshd on 127.0.0.1:SSH_PORT for the launcher's key only."""
    if not CFG["ssh_pubkey"]:
        return False
    sshd = apt_install("openssh-server", "sshd")
    if not sshd:
        log("   no sshd -> SSH channel off")
        return False
    home = Path(os.path.expanduser("~"))
    (home / ".ssh").mkdir(mode=0o700, exist_ok=True)
    keys = home / ".ssh" / "authorized_keys"
    have = keys.read_text() if keys.exists() else ""
    if CFG["ssh_pubkey"].strip() not in have:     # keep keys other tools (DeployMan) put there
        keys.write_text(have + ("" if not have or have.endswith("\n") else "\n") + CFG["ssh_pubkey"].strip() + "\n")
    keys.chmod(0o600)
    Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh-keygen", "-A"], capture_output=True)
    log_fh = open(WORK / "sshd.log", "a")
    subprocess.Popen([sshd, "-D", "-e", "-p", str(SSH_PORT), "-o", "ListenAddress=127.0.0.1",
                      "-o", "PasswordAuthentication=no", "-o", "PermitRootLogin=prohibit-password",
                      "-o", "AllowTcpForwarding=yes", "-o", "ClientAliveInterval=30"],
                     stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    for _ in range(20):
        if port_open(SSH_PORT):
            return True
        time.sleep(0.5)
    log("   sshd did not start (see sshd.log)")
    return False


def open_tunnels(wait_s=150):
    starters = [("ngrok", start_ngrok), ("cloudflared", start_cloudflared), ("pinggy", start_pinggy)]

    def ssh_channel():
        # own thread: installing sshd can take minutes and must not delay the API tunnels
        try:
            if not start_sshd():
                return
            for name, start in (("cloudflared-ssh", lambda: start_cloudflared(
                                    "cloudflared-ssh", f"ssh://127.0.0.1:{SSH_PORT}")),
                                ("pinggy-ssh", lambda: start_pinggy("pinggy-ssh", SSH_PORT, tcp=True))):
                if name not in TUNNELS:
                    start()
            start_watchdog()
            time.sleep(60)
            publish("ssh-channel", tunnels=tunnel_summary())
        except Exception as e:  # noqa: BLE001
            log(f"   ssh channel error: {e}")
    if CFG["ssh_pubkey"] and not SSH_BOOT:
        SSH_BOOT.append(threading.Thread(target=ssh_channel, daemon=True))
        SSH_BOOT[0].start()
    for name, start in starters:
        if name in TUNNELS:       # already started from the command channel
            continue
        try:
            start()
        except Exception as e:  # noqa: BLE001
            log(f"   tunnel start error: {e}")
    if not TUNNELS:
        publish("tunnel-failed", note=f"no tunnel could be started; server only on :{PORT} "
                                      "inside the kernel (command channel still works)")
        return None
    t0 = time.time()
    while time.time() < t0 + wait_s and any(
            t["state"] == "starting" and t["proc"].poll() is None for t in list(TUNNELS.values())):
        # one working tunnel is enough to announce; don't wait out a blocked one
        if time.time() > t0 + 45 and any(t["state"] == "online" for t in list(TUNNELS.values())):
            break
        time.sleep(2)
    time.sleep(3)
    for t in list(TUNNELS.values()):
        log(f"   {t['name']:<15} {t['url'] or '-'}  -> {probe_tunnel(t)}")
    start_watchdog()
    url = primary_url()
    publish("tunnel-url", endpoint=f"{url}/v1" if url else None, tunnels=tunnel_summary())
    return url


# ---------------- commands (over /ktl/cmd or ntfy) ----------------
HELP = """commands:
  status             server + tunnel state
  log [N]            last N lines of the server log (default 60)
  tunnel [restart]   probe tunnels / restart them
  tunnel pinggy      (re)start the pinggy tunnel (port 443, no account)
  tunnel ngrok <url> <authtoken>   start ngrok with your static domain right now
  sh <command>       run a shell command (unknown words run as shell too)
  stop               stop the server and end the Kaggle session"""


def status_text():
    code, _ = http_get(f"http://127.0.0.1:{BACKEND_PORT}/v1/models",
                       {"Authorization": f"Bearer {CFG['api_key']}"}, timeout=5)
    server = "?"
    try:
        server = HOOKS["server_status"]() if HOOKS["server_status"] else \
            ("listening" if port_open(BACKEND_PORT) else "not listening")
    except Exception as e:  # noqa: BLE001
        server = f"status error: {e}"
    lines = [f"uptime {elapsed()}, last phase: {STATE['phase']}",
             f"server: {server}, local HTTP: {code}"]
    for n, t in list(TUNNELS.items()):
        lines.append(f"{n}: {t['url']}  state={t['state']}  probe={t['probe']}  "
                     f"restarts={t['restarts']}  {t['note']}")
    return "\n".join(lines)


def send_result(ref, cmd, rc, out):
    out = out or "(no output)"
    if len(out) > 12000:
        out = "...[truncated]\n" + out[-12000:]
    parts, i = [], 0
    while i < len(out):
        n = 3000
        while n > 200 and len(json.dumps(out[i:i + n])) > 3300:
            n //= 2
        parts.append(out[i:i + n])
        i += n
    for k, chunk in enumerate(parts):
        ntfy_send({"phase": "cmd-result", "ref": ref, "cmd": cmd[:200], "rc": rc,
                   "part": k, "parts": len(parts), "out": chunk})


def run_command(text, ref):
    """ntfy path: acknowledge, execute, reply in chunks on the progress topic."""
    text = text.strip()
    shown = re.sub(r"^(tunnel ngrok \S+) \S+", r"\1 <token>", text)   # don't echo secrets
    ntfy_send({"phase": "cmd-ack", "ref": ref, "cmd": shown[:200]})
    if text.partition(" ")[0] == "stop":
        send_result(ref, shown, 0, "stopping the server and ending the session")
    rc, out = execute(text)
    send_result(ref, shown, rc, out)


def execute(text):
    """Run one control command. Returns (rc, output)."""
    text = text.strip()
    word, _, rest = text.partition(" ")
    rc = 0
    try:
        if word in ("help", "?"):
            out = HELP
        elif word == "status":
            out = status_text()
        elif word == "log":
            out = log_tail(int(rest) if rest.strip().isdigit() else 60)
        elif word == "tunnel":
            sub = rest.split()
            if sub[:1] == ["restart"]:
                for t in list(TUNNELS.values()):
                    t["proc"].terminate()
                start_watchdog()
                out = "tunnels terminated; the watchdog restarts them within ~30 s"
            elif sub[:1] in (["pinggy"], ["ngrok"]):
                name = sub[0]
                if name == "ngrok":
                    if len(sub) < 3:
                        raise ValueError("usage: tunnel ngrok <https://name.ngrok-free.dev> <authtoken>")
                    CFG["ngrok_url"], CFG["ngrok_authtoken"] = sub[1], sub[2]
                else:
                    CFG["pinggy"] = True
                old = TUNNELS.pop(name, None)
                if old and old["proc"].poll() is None:
                    old["proc"].terminate()
                    time.sleep(2)
                t = (start_ngrok if name == "ngrok" else start_pinggy)()
                if t is None:
                    raise RuntimeError(f"{name} could not be started (see `log 40`)")
                start_watchdog()
                for _ in range(30):
                    if t["url"] and t["state"] != "starting":
                        break
                    time.sleep(2)
                out = f"{name}: {t['url']} -> {probe_tunnel(t)}"
                publish("tunnel-restart", name=name, restarts=t["restarts"],
                        endpoint=f"{t['url']}/v1" if t["url"] else None, status=t["probe"])
            else:
                out = "\n".join(f"{n}: {t['url']} -> {probe_tunnel(t)}"
                                for n, t in list(TUNNELS.items())) or "no tunnels"
        elif word == "stop":
            publish("stopped", reason="stop-command")
            threading.Timer(2, shutdown, args=(0,)).start()
            out = "stopping the server and ending the session"
        else:
            p = subprocess.run(["bash", "-lc", rest if word == "sh" else text],
                               capture_output=True, text=True, timeout=CMD_TIMEOUT, cwd=str(WORK))
            rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        partial = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, out = 124, f"timed out after {CMD_TIMEOUT} s\n{partial}"
    except Exception as e:  # noqa: BLE001
        rc, out = 1, f"error: {e}"
    return rc, out


def cmd_listener():
    topic, since = CFG["ntfy_cmd_topic"], str(int(T0) - 60)
    while True:
        try:
            with urllib.request.urlopen(f"https://ntfy.sh/{topic}/json?since={since}", timeout=120) as r:
                for line in r:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("event") != "message":
                        continue
                    since = e["id"]
                    threading.Thread(target=run_command, args=(e.get("message", ""), e["id"]),
                                     daemon=True).start()
        except Exception:  # noqa: BLE001
            time.sleep(10)


# ---------------- front server: API proxy + control over any tunnel ----------------
HOP_HEADERS = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te",
               "trailer", "upgrade", "host", "content-length"}


class Front(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"         # closing the connection ends a streamed body

    def log_message(self, fmt, *args):
        pass

    def _authorized(self):
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        token = token or self.headers.get("X-KTL-Token", "") or self.headers.get("x-api-key", "")
        return bool(token) and any(secrets.compare_digest(token, good) for good in
                                   (CFG["api_key"], CFG["ntfy_cmd_topic"]) if good)

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _control(self):
        path = self.path.split("?", 1)[0]
        if path == "/ktl/health":
            return self._json(200, {"ktl": True, "server": port_open(BACKEND_PORT),
                                    "phase": STATE["phase"], "uptime": elapsed()})
        if not self._authorized():
            return self._json(401, {"error": "bearer token required"})
        if path == "/ktl/tunnels":
            return self._json(200, {"tunnels": tunnel_summary(), "phase": STATE["phase"]})
        if path == "/ktl/cmd" and self.command == "POST":
            rc, out = execute(self._body().decode("utf-8", "replace"))
            return self._json(200, {"rc": rc, "out": out})
        return self._json(404, {"error": "unknown control route"})

    def _forward(self):
        body = self._body()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=3600)
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except OSError:
            conn.close()
            return self._json(503, {"error": {"type": "server_unavailable", "code": 503,
                                              "message": "the model server is not up "
                                              f"(phase: {STATE['phase']}); retry later"}})
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_HEADERS or (k.lower() == "content-length" and not resp.chunked):
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            pass
        finally:
            conn.close()

    def _handle(self):
        try:
            if self.path.startswith("/ktl/"):
                self._control()
            else:
                self._forward()
        except Exception as e:  # noqa: BLE001  never let one request kill the front
            raw(f"[front] {self.command} {self.path}: {e!r}")

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _handle


def start_front():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Front)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start(tunnels=True):
    """Front server now; tunnels + SSH in the background. Returns the boot thread (join it to wait for URLs)."""
    if CFG["ntfy_cmd_topic"]:
        threading.Thread(target=cmd_listener, daemon=True).start()
    BOOT["front"] = start_front()
    if tunnels:
        BOOT["thread"] = threading.Thread(target=lambda: BOOT.__setitem__("url", open_tunnels()), daemon=True)
        BOOT["thread"].start()
    return BOOT.get("thread")


def run_net_test(keepalive_min):
    """CPU smoke test: a dummy backend behind the front server, the tunnels and SSH, then hold."""
    log("=" * 70)
    log(f" NETWORK TEST — front :{PORT} -> dummy HTTP server :{BACKEND_PORT} + tunnels + channels")
    log("=" * 70)
    dummy = subprocess.Popen([sys.executable, "-m", "http.server", str(BACKEND_PORT), "--bind", "127.0.0.1"],
                             cwd=str(WORK), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    HOOKS["server_status"] = lambda: "running" if dummy.poll() is None else f"exited rc={dummy.returncode}"
    if BOOT.get("thread"):
        BOOT["thread"].join(timeout=240)
    summary = probe_all()
    publish("nettest-result", ok=any(v["status"] == "live" for v in summary.values()),
            tunnels=summary, command_channel=bool(CFG["ntfy_cmd_topic"]),
            ssh=bool(CFG["ssh_pubkey"]), arch=platform.machine(), keepalive_min=keepalive_min)
    t_end = time.time() + keepalive_min * 60
    while time.time() < t_end:
        time.sleep(30)
    publish("auto-shutdown", served_min=keepalive_min)
    shutdown(0)
