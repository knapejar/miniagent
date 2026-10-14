# -*- coding: utf-8 -*-
"""Configuration - one place for every default.

A profile is one model on one backend. miniagent ships a single one, "local":
Spark-X2.5-4B in LM Studio on this machine. Every other profile is your own, a
JSON file per provider in ~/.miniagent/profiles/ - the file's name is the
provider, its top-level keys are shared by all of its profiles:

    ~/.miniagent/profiles/hosted.json
    {
      "url": "https://api.example.com/v1",
      "api_key_env": "EXAMPLE_API_KEY",
      "ctx": 131072,
      "profiles": {
        "big":   {"model": "some-model-70b"},
        "small": {"model": "some-model-8b", "effort": "medium"}
      }
    }

A profile's own keys win over the file's, the command line wins over both, and
"<key>_env" names an environment variable that replaces <key> when it is set.
The keys are those of DEFAULTS; a connector may read more (see agent/connectors).
"""
import argparse
import glob
import json
import os

from . import paths
from .protocol import DIALECTS, dialect_for
from .secrets import register

EFFORTS = ("default", "none", "low", "medium", "high", "xhigh")

DEFAULTS = {
    # What brings the backend up before the first step: lmstudio (also any plain
    # OpenAI-compatible URL, which it only checks), llamacpp or kaggle.
    "connector": "lmstudio",
    "url": "http://127.0.0.1:1234/v1",
    "model": "",
    "api_key": "",
    "headers": {},                 # extra HTTP headers a provider asks for
    "dialect": "auto",             # tool-call format, see protocol.py
    "ctx": 65536,
    "max_tokens": 8192,
    "temperature": None,           # None = not sent, the server's own default
    "top_p": None,
    "top_k": None,
    "effort": "low",
    # Send the effort as chat_template_kwargs in Qwen3.8's levels instead of
    # reasoning_effort - what vLLM serving Qwen3.8 needs, see llm.py.
    "effort_in_template": False,
    "retries": 0,                  # another try on a tunnel/proxy error
    "outage_wait": 0,              # seconds to wait out a server outage
}

LOCAL = {
    "url": "http://127.0.0.1:1234/v1", "url_env": "MINIAGENT_URL",
    "model": "spark-x2.5-4b", "model_env": "MINIAGENT_MODEL",
    "api_key": "lmstudio", "api_key_env": "MINIAGENT_KEY",
    # Spark-X2.5 official recommendation: temperature 1.0, top_p 0.95, top_k -1.
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": -1,
    # LM Studio ignores chat_template_kwargs; the only working switch for
    # reasoning is reasoning_effort. On large tasks the model loops forever
    # inside <think>, so reasoning is low by default.
    "effort": "low",
}


def load_profiles():
    """{name: fields} - the built-in local, then ~/.miniagent/profiles/*.json.
    Each profile carries "provider", the name of the file it came from."""
    found = {"local": dict(LOCAL, provider="lmstudio")}
    for path in sorted(glob.glob(os.path.join(paths.profiles_dir(), "*.json"))):
        try:
            with open(path, encoding="utf-8-sig") as fh:
                group = json.load(fh)
            shared = {k: v for k, v in group.items() if k != "profiles"}
            shared["provider"] = os.path.splitext(os.path.basename(path))[0]
            for name, fields in group.get("profiles", {}).items():
                found[name] = dict(shared)
                found[name].update(fields)
        except (OSError, ValueError, AttributeError) as e:
            raise ValueError("bad profile file %s: %s" % (path, e))
    return found


def resolve(fields):
    """A profile's fields with every "<key>_env" that is set applied."""
    out = dict(fields)
    for key, env in fields.items():
        if key.endswith("_env") and os.environ.get(env):
            out[key[:-4]] = os.environ[env]
    return out


def read_key(name):
    """A key kept in ~/.miniagent/<name>.key, for backends that need one and
    an environment variable that a new terminal would have to be opened for."""
    try:
        # utf-8-sig: a key file written by PowerShell carries a BOM, and a BOM
        # in an HTTP header is a UnicodeEncodeError, not a bad key.
        with open(paths.key_file(name), encoding="utf-8-sig") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def stored_default():
    """MINIAGENT_PROFILE, else what `--profile <name> --set-default` wrote, else ""."""
    name = os.getenv("MINIAGENT_PROFILE")
    if not name:
        try:
            with open(paths.default_profile_file(), encoding="utf-8") as fh:
                name = fh.read().strip()
        except OSError:
            name = ""
    return name if name in load_profiles() else ""


def set_default_profile(name):
    path = paths.default_profile_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(name + "\n")
    return path


class Config(object):
    """Everything that shapes a run. Built from command line arguments.

    profile=None leaves the model unpicked: the REPL starts without one and
    /model chooses it (use_profile)."""

    def __init__(self, **kw):
        self.workdir = kw.get("workdir", ".")
        self.max_steps = kw.get("max_steps", 60)
        # Share of max_tokens the reasoning may take before the reply is cut and
        # asked for again with less thinking; 0 turns the cap off.
        self.think_cap = kw.get("think_cap", 0.6)
        # One effort level more where thinking pays: the first step, and the step
        # after the code failed to run (loop.py).
        self.think_boost = kw.get("think_boost", False)

        self.crop_at = kw.get("crop_at", 0.7)      # share of ctx that triggers cropping
        self.keep_tail = kw.get("keep_tail", 12)   # messages the crop keeps at the end
        # A plan item turning DONE starts the next one in a clean context, once the
        # run has grown past this share of ctx; 0 turns it off (loop.py).
        self.checkpoint_at = kw.get("checkpoint_at", 0.3)
        self.remind_every = kw.get("remind_every", 6)

        self.approve = kw.get("approve", "none")   # auto | all | none
        self.stream = kw.get("stream", True)
        self.color = kw.get("color", True)
        self.show_reasoning = kw.get("show_reasoning", True)
        self.redact = kw.get("redact", True)       # mask secrets in tool output
        self.async_web = kw.get("async_web", False)  # never block on web tools
        # After the answer, one short run that updates the project's memory.
        self.memory = kw.get("memory", True)
        # Traces go to ~/.miniagent/projects/<slug>/runs, next to the memory for
        # the same project, so no run ever litters the directory it works in.
        self.trace_dir = kw.get("trace_dir") or paths.runs_dir(self.workdir)
        self.run_name = kw.get("run_name", None)
        self.timeout = kw.get("timeout", 900)

        # kaggle connector only
        self.spawn_proxy = kw.get("spawn_proxy", True)   # start launch.py proxy if needed
        self.wait = kw.get("wait", False)                # wait until the model answers
        self.use_profile(kw.get("profile", "local"), kw)

    def use_profile(self, name, given=None):
        """Point at a profile's model; anything in `given` wins over the profile.
        name=None means no model picked yet."""
        given = given or {}
        fields = resolve(load_profiles()[name]) if name else {}

        def pick(key):
            if given.get(key) is not None:
                return given[key]
            return fields.get(key, DEFAULTS[key])

        self.profile = name
        self.provider = fields.get("provider", "")
        self.options = fields                  # what a connector reads beyond DEFAULTS
        self.connector = pick("connector")
        self.url = pick("url")
        self.model = pick("model")
        self.api_key = pick("api_key")
        if not self.api_key and name:
            # ~/.miniagent/<profile>.key, else one key for the provider's profiles
            self.api_key = read_key(name) or read_key(self.provider)
            register([self.api_key])
        self.headers = pick("headers")
        dialect = pick("dialect")
        self.dialect = dialect_for(self.model) if dialect == "auto" else dialect
        self.ctx = pick("ctx")
        self.max_tokens = pick("max_tokens")
        self.temperature = pick("temperature")
        self.top_p = pick("top_p")
        self.top_k = pick("top_k")
        self.effort = pick("effort")
        self.effort_in_template = pick("effort_in_template")
        self.retries = pick("retries")
        self.outage_wait = pick("outage_wait")
        self.ready = False                     # set once the connector got an answer
        self.backend_note = ""                 # one banner line, set by the connector

    def as_dict(self):
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and k not in ("api_key", "options", "headers")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="miniagent",
        description="An agentic CLI for local models. Without --profile the "
                    "interactive mode starts with no model; /model picks one.")
    p.add_argument("task", nargs="*", help="the task; omit it for interactive mode")
    p.add_argument("-f", "--task-file", help="read the task from a file")
    p.add_argument("-C", "--workdir", default=".", help="working directory for the agent")

    g = p.add_argument_group("model")
    g.add_argument("--profile", nargs="?", const="", default=None, metavar="NAME",
                   help="the model to use (local, or one of ~/.miniagent/profiles); "
                        "without a name, list them")
    g.add_argument("--set-default", action="store_true",
                   help="remember --profile as the default for next time")
    g.add_argument("--url", help="OpenAI-compatible endpoint")
    g.add_argument("--model", help="model identifier")
    g.add_argument("--api-key")
    g.add_argument("--dialect", choices=("auto",) + DIALECTS, default="auto",
                   help="tool-call format: spark | qwen (default: from the model name)")
    g.add_argument("--effort", choices=EFFORTS, help="reasoning effort")
    g.add_argument("--think-boost", action="store_true",
                   help="one effort level more on the first step and after a failing run")
    g.add_argument("--think-cap", type=float, metavar="SHARE",
                   help="cut a reply whose reasoning passes this share of "
                        "--max-tokens (default 0.6, 0 = off)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--ctx", type=int)

    g = p.add_argument_group("kaggle connector")
    g.add_argument("--no-proxy", action="store_true",
                   help="do not start `launch.py proxy` when nothing answers on the URL")
    g.add_argument("--wait", action="store_true",
                   help="wait until the Kaggle model is live before starting")

    g = p.add_argument_group("loop")
    g.add_argument("--max-steps", type=int, default=60)
    g.add_argument("--crop-at", type=float, default=0.7)
    g.add_argument("--keep-tail", type=int, default=12)
    g.add_argument("--checkpoint-at", type=float, default=0.3, metavar="SHARE",
                   help="when a plan item turns DONE past this share of the context, "
                        "drop what it took to get there (0 = never)")
    g.add_argument("--remind-every", type=int, default=6)
    g.add_argument("--approve", choices=["auto", "all", "none"], default=None,
                   help="auto | all | none (default: none = run everything)")

    g = p.add_argument_group("output")
    g.add_argument("--no-stream", action="store_true", help="disable streaming")
    g.add_argument("--no-color", action="store_true")
    g.add_argument("--no-reasoning", action="store_true",
                   help="do not print the model's reasoning (shown live by default)")
    g.add_argument("--no-redact", action="store_true", help="do not mask secrets in output")
    g.add_argument("--async-web", action="store_true",
                   help="websearch and browse never block; results arrive later")
    g.add_argument("--no-memory", action="store_true",
                   help="skip the memory pass that runs after the answer")
    g.add_argument("--run-name", help="name of the trace file")
    g = p.add_argument_group("headless (the same flags as Claude Code's print mode)")
    g.add_argument("-p", "--print", action="store_true",
                   help="run without the interactive UI and print the result")
    g.add_argument("--output-format", choices=("text", "json", "stream-json"), default="text")
    g.add_argument("--input-format", choices=("text", "stream-json"), default="text",
                   help="stream-json: one user message per stdin line, the process "
                        "keeps the conversation until stdin closes")
    g.add_argument("--include-partial-messages", action="store_true",
                   help="stream text and thinking deltas as stream_event lines")
    g.add_argument("--verbose", action="store_true", help="accepted for compatibility")
    g.add_argument("--session-id", metavar="UUID", help="id to save this conversation under")
    g.add_argument("--resume", "-r", metavar="ID", help="continue a saved conversation by id")
    g.add_argument("--append-system-prompt", metavar="TEXT", default="",
                   help="added to the end of the system prompt")
    g.add_argument("--append-system-prompt-file", metavar="FILE",
                   help="the same, read from a file")
    g.add_argument("--mcp-config", action="append", metavar="JSON|FILE",
                   help="MCP servers ({\"mcpServers\": {...}}); may be repeated")
    g.add_argument("--strict-mcp-config", action="store_true",
                   help="accepted for compatibility (no other MCP config is read)")
    g.add_argument("--dangerously-skip-permissions", action="store_true",
                   help="run every tool without asking (same as --approve none)")
    g.add_argument("--disallowedTools", "--disallowed-tools", nargs="*", default=[],
                   metavar="TOOL", help="tools the model is not offered")
    p.add_argument("-v", "--version", action="version", version="miniagent 0.2.0")

    a = p.parse_args(argv)
    try:
        profiles = load_profiles()
        if a.profile == "":                   # --profile with no name: just list them
            from .connectors import profiles_text
            p.exit(0, profiles_text() + "\n")
        default = stored_default()
    except ValueError as e:
        p.error(str(e))
    task = " ".join(a.task)
    if not task and a.task_file:
        with open(a.task_file, encoding="utf-8") as fh:
            task = fh.read().strip()
    # Interactive with nothing chosen starts without a model. A one-shot task has
    # nobody to ask, and --url/--model already say which model: both build on
    # local, as they always did.
    profile = a.profile or default or ("local" if task or a.url or a.model or a.print else None)
    if profile is not None and profile not in profiles:
        p.error("unknown profile %r; --profile lists them" % profile)
    if a.set_default and profile:
        print("  default profile: %s (%s)" % (profile, set_default_profile(profile)))
    given = {name: getattr(a, name) for name in
             ("url", "model", "api_key", "effort", "ctx", "max_tokens",
              "temperature", "top_p", "top_k", "think_cap", "think_boost")
             if getattr(a, name) is not None}
    if a.dialect != "auto":
        given["dialect"] = a.dialect
    cfg = Config(profile=profile,
                 workdir=a.workdir, max_steps=a.max_steps,
                 crop_at=a.crop_at, keep_tail=a.keep_tail, remind_every=a.remind_every,
                 checkpoint_at=a.checkpoint_at,
                 approve="none" if a.dangerously_skip_permissions else (a.approve or "none"),
                 stream=not a.no_stream, color=not a.no_color,
                 show_reasoning=not a.no_reasoning, redact=not a.no_redact,
                 async_web=a.async_web, memory=not a.no_memory, run_name=a.run_name,
                 spawn_proxy=not a.no_proxy, wait=a.wait, **given)
    cfg.append_system = a.append_system_prompt
    if a.append_system_prompt_file:
        with open(a.append_system_prompt_file, encoding="utf-8-sig") as fh:
            cfg.append_system = (cfg.append_system + "\n\n" + fh.read()).strip()
    cfg.headless = None
    if a.print:
        cfg.headless = {"output_format": a.output_format, "input_format": a.input_format,
                        "partial": a.include_partial_messages, "session_id": a.session_id,
                        "resume": a.resume, "mcp_config": a.mcp_config or [],
                        "disallowed": a.disallowedTools}
    return cfg, task
