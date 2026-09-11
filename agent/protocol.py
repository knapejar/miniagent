# -*- coding: utf-8 -*-
"""Tool-call protocol.

Spark-X2.5 calls tools with its own text format:

    <tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

llama.cpp has no parser for it (PR #27868 added the architecture only), so we
deliberately do NOT send the OpenAI `tools` field: that would trigger the Hermes
parser and its grammar would push the model into JSON it was never trained on.
Instead we render the tool specs into the system prompt exactly the way the
model's own chat template does, and parse the reply ourselves.
"""
import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

PLAN_HEADER = ("[PLAN - your own notes, pinned here and kept across context cropping. "
               "Update it with the plan tool whenever a part starts working.]\n")
PLAN_MAX_CHARS = 1200

SYSTEM_TEMPLATE = """You are an autonomous agent driving a {os_name} machine through {shell}.

## Tools
You have access to the following functions:
<tools>
{tools}
</tools>

Rules:
- Emit exactly ONE <tool_call> per message, then stop and wait for the <tool_response>.
- Never invent tool output. Decide only from what the response actually says.
- Keep output small: grep instead of dumping files, dir /b, narrow read ranges.
- Verify your work with a tool before claiming success.
- Keep the plan updated: what already works must be marked so you do not break it.
- When the task is finished, reply with a plain answer of AT MOST 5 lines and no
  tool_call. No status tables, no checklists, no restating what you did.
- If you are blocked by something you cannot do yourself, call ask instead of
  retrying a command that already failed.

Working directory: {cwd}{secrets}"""


def render_tools(tools):
    """Render tool specs the way the model's chat template would."""
    return "\n".join(json.dumps(t.spec(), ensure_ascii=False, separators=(",", ":"))
                     for t in tools)


def build_system(tools, cwd, os_name="Windows 11", shell="cmd.exe", secret_names=()):
    secrets = ""
    if secret_names:
        secrets = ("\nSecrets available as environment variables, referenced as %NAME% in sh. "
                   "Their values are never shown to you and must never be typed out: "
                   + ", ".join(sorted(secret_names)))
    return SYSTEM_TEMPLATE.format(tools=render_tools(tools), cwd=cwd,
                                  os_name=os_name, shell=shell, secrets=secrets)


def strip_think(text):
    """Drop reasoning. The model's template opens generation inside <think>, so
    a reply often carries a closing tag with no opening one."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return THINK_RE.sub("", text).strip()


def parse_tool_call(text):
    """Return (name, args) or None.

    Handles the native Spark format, Hermes JSON, and an unterminated call
    (generation stopped on the stop sequence).
    """
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    body = match.group(1)
    pairs = ARG_RE.findall(body)
    if pairs:
        name = body.split("<arg_key>", 1)[0].strip().strip('"')
        return name, _merge(pairs)
    try:
        obj = json.loads(body.strip())
        return obj.get("name"), obj.get("arguments") or obj.get("parameters") or {}
    except Exception:
        stripped = body.strip()
        return (stripped.split()[0], {}) if stripped else None


def _merge(pairs):
    """A repeated key is several values, not the last one. The model expresses
    "search for these four things" as four <arg_key>query</arg_key> pairs, and
    collapsing them into a dict silently threw three of them away."""
    args = {}
    for key, value in pairs:
        key = key.strip()
        args[key] = args[key] + "\n" + value if key in args else value
    return args


MARKUP = ("<arg_key>", "<arg_value>", "<tool_call>")


def malformed(args):
    """Tool-call markup left inside an argument value means the call was nested
    or otherwise broken. Executing it runs a fragment: one such call reached the
    shell as a command and came back as "< was unexpected at this time.", which
    tells the model nothing about what it did wrong."""
    return any(marker in str(value) for value in (args or {}).values()
               for marker in MARKUP)


def format_tool_call(name, args):
    """Inverse of parse_tool_call - useful for tests and replaying history."""
    parts = ["<tool_call>", name]
    for key, value in args.items():
        parts.append("<arg_key>%s</arg_key><arg_value>%s</arg_value>" % (key, value))
    parts.append("</tool_call>")
    return "".join(parts)


def wrap_tool_response(text):
    return "<tool_response>%s</tool_response>" % text
