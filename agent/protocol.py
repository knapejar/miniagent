# -*- coding: utf-8 -*-
"""Tool-call protocol.

Spark-X2.5 calls tools with its own text format:

    <tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

llama.cpp has no parser for it (PR #27868 added the architecture only), so we
deliberately do NOT send the OpenAI `tools` field: that would trigger the Hermes
parser and its grammar would push the model into JSON it was never trained on.
Instead we render the tool specs into the system prompt exactly the way the
model's own chat template does, and parse the reply ourselves.

Other models were trained on other shapes. Nanbeige4.2 writes

    <tool_call><function=name><parameter=key>value</parameter></function></tool_call>

and, given Spark's prompt, mixes the two into `<function{name": "sh", ...}}`,
which parses into nothing. So the system prompt carries a per-model call-format
block (DIALECTS, chosen by model name) while the parser stays permissive and
reads every shape it knows - a model that drifts mid-run is still understood.

Everything else gets the standard OpenAI function calling ("openai" dialect):
the specs go into the request's `tools` field, the server's own parser (vLLM
qwen3_coder, LM Studio, a hosted API...) returns structured `tool_calls`, and
results travel back as role "tool" messages carrying the call id. That is the
only shape a remote endpoint such as Qwen3.8-27B on vLLM is guaranteed to parse.
"""
import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
FUNC_RE = re.compile(r"<function=(.*?)>(.*)", re.S)
# The terminator is a lookahead, not a closing tag: a call cut off by the stop
# sequence keeps its last argument instead of losing every one of them.
PARAM_RE = re.compile(
    r"<parameter=(.*?)>\s*(.*?)\s*(?=</parameter>|<parameter=|</function>|$)", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

# Copied word for word from each model's own chat template: the closer the
# prompt is to what the model saw in training, the cleaner the calls come back.
NANBEIGE_FORMAT = """
For each function call, output the function name and arguments within the following XML format:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
"""

# Spark needs no instruction - its own format is what it reaches for unprompted.
DIALECTS = {"spark": "", "nanbeige": NANBEIGE_FORMAT, "openai": ""}
NATIVE = "openai"


def dialect_for(model):
    """Pick the tool-call dialect from the model name: the text formats for the
    small local models they were tuned on, standard function calling otherwise."""
    name = (model or "").lower()
    if "nanbeige" in name:
        return "nanbeige"
    if not name or "spark" in name:
        return "spark"
    return NATIVE


PLAN_HEADER = ("[PLAN - your own notes, pinned here and kept across context cropping. "
               "Update it with the plan tool whenever a part starts working.]\n")
PLAN_MAX_CHARS = 1200

SYSTEM_TEMPLATE = """You are an autonomous agent driving a {os_name} machine through {shell}.

## Tools
You have access to the following functions:
<tools>
{tools}
</tools>
{call_format}
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

# The server renders the tool specs itself, so they are not repeated here.
NATIVE_SYSTEM_TEMPLATE = """You are an autonomous agent driving a {os_name} machine through {shell}.

Rules:
- Call exactly ONE tool per message, then stop and wait for its result.
- Never invent tool output. Decide only from what the result actually says.
- Keep output small: grep instead of dumping files, dir /b, narrow read ranges.
- Verify your work with a tool before claiming success.
- Keep the plan updated: what already works must be marked so you do not break it.
- When the task is finished, reply with a plain answer of AT MOST 5 lines and no
  tool call. No status tables, no checklists, no restating what you did.
- If you are blocked by something you cannot do yourself, call ask instead of
  retrying a command that already failed.

Working directory: {cwd}{secrets}"""


def render_tools(tools):
    """Render tool specs the way the model's chat template would."""
    return "\n".join(json.dumps(t.spec(), ensure_ascii=False, separators=(",", ":"))
                     for t in tools)


def tools_field(tools):
    """Tool specs for the request's `tools` field (the "openai" dialect)."""
    return [{"type": "function", "function": t.spec()} for t in tools]


def native_args(arguments):
    """Arguments of a structured tool call -> dict of strings, the same shape the
    text dialects produce, so every tool sees what it always saw. Returns None
    when the server passed something that is not a JSON object."""
    if isinstance(arguments, dict):
        obj = arguments
    else:
        try:
            obj = json.loads(arguments or "{}")
        except ValueError:
            return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for key, value in obj.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            out[key] = str(value)
        elif value is not None:
            out[key] = json.dumps(value, ensure_ascii=False)
    return out


def build_system(tools, cwd, os_name="Windows 11", shell="cmd.exe", secret_names=(),
                 dialect="spark"):
    secrets = ""
    if secret_names:
        secrets = ("\nSecrets available as environment variables, referenced as %NAME% in sh. "
                   "Their values are never shown to you and must never be typed out: "
                   + ", ".join(sorted(secret_names)))
    if dialect == NATIVE:
        return NATIVE_SYSTEM_TEMPLATE.format(cwd=cwd, os_name=os_name, shell=shell,
                                             secrets=secrets)
    return SYSTEM_TEMPLATE.format(tools=render_tools(tools), cwd=cwd,
                                  os_name=os_name, shell=shell, secrets=secrets,
                                  call_format=DIALECTS.get(dialect, ""))


def strip_think(text):
    """Drop reasoning. The model's template opens generation inside <think>, so
    a reply often carries a closing tag with no opening one."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return THINK_RE.sub("", text).strip()


def parse_tool_call(text):
    """Return (name, args) or None.

    Handles the native Spark format, Nanbeige's <function=>/<parameter=>,
    Hermes JSON, and an unterminated call (generation stopped on the stop
    sequence).
    """
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    body = match.group(1)
    pairs = ARG_RE.findall(body)
    if pairs:
        name = body.split("<arg_key>", 1)[0].strip().strip('"')
        return name, _merge(pairs)
    func = FUNC_RE.search(body)
    if func:
        return func.group(1).strip(), _merge(PARAM_RE.findall(func.group(2)))
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


def format_tool_call(name, args, dialect="spark"):
    """Inverse of parse_tool_call - useful for tests and replaying history."""
    if dialect == "nanbeige":
        parts = ["<tool_call>", "<function=%s>" % name]
        for key, value in args.items():
            parts.append("<parameter=%s>\n%s\n</parameter>" % (key, value))
        parts.append("</function>")
        parts.append("</tool_call>")
        return "".join(parts)
    parts = ["<tool_call>", name]
    for key, value in args.items():
        parts.append("<arg_key>%s</arg_key><arg_value>%s</arg_value>" % (key, value))
    parts.append("</tool_call>")
    return "".join(parts)


def wrap_tool_response(text):
    return "<tool_response>%s</tool_response>" % text
