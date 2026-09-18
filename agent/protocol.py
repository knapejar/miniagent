# -*- coding: utf-8 -*-
"""Tool-call protocol.

Spark-X2.5 calls tools with its own text format:

    <tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

llama.cpp has no parser for it (PR #27868 added the architecture only), so we
deliberately do NOT send the OpenAI `tools` field: that would trigger the Hermes
parser and its grammar would push the model into JSON it was never trained on.
Instead we render the tool specs into the system prompt exactly the way the
model's own chat template does, and parse the reply ourselves.

Qwen3.8 (served by vLLM from kaggle-tpu-lab) has a different native format, the
qwen3_coder XML:

    <tool_call>
    <function=name>
    <parameter=key>
    value
    </parameter>
    </function>
    </tool_call>

The server has a parser for it, but it only runs when the request carries
`tools`, so the same approach holds: the "qwen" dialect writes the format into
the system prompt the way Qwen's template does, and the parser reads both.
"""
import json
import os
import re

from . import paths

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>")
# A value ends at its closing tag - or, when the model forgot it, at the next
# parameter, the end of the function, or the end of the text.
PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.S)
DIALECTS = ("spark", "qwen")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

PLAN_HEADER = ("[PLAN - your own notes, pinned here and kept across context cropping. "
               "Update it with the plan tool whenever a part starts working.]\n")
PLAN_MAX_CHARS = 1200

SPARK_TOOLS = """## Tools
You have access to the following functions:
<tools>
{tools}
</tools>"""

# Worded after Qwen's own chat template: staying close to what the model saw in
# training is what makes it emit the format reliably without a server parser.
QWEN_TOOLS = """# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

To call a function, reply in this format with NO text after it:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

Parameter values are plain text: no quotes, no JSON escaping, file content verbatim."""

SYSTEM_TEMPLATE = """You are an autonomous agent driving a {os_name} machine through {shell}.

{tools_block}

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
- Your state for this project lives in {state_dir}: memory/INDEX.md (one line per
  note, nest a subfolder with its own INDEX.md per topic once one file is not
  enough) and runs/*.jsonl (the trace of every past run here). Read a note, or
  grep the old runs, before rediscovering something; write a note only for a fact
  worth reusing later (a host, a login, a working command) - create it if missing.

Working directory: {cwd}{secrets}{memory}"""

MEMORY_MAX_CHARS = 1500


def _load_memory(cwd):
    path = paths.memory_index(cwd)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError:
        return ""
    if not text:
        return ""
    if len(text) > MEMORY_MAX_CHARS:
        text = text[:MEMORY_MAX_CHARS] + "\n... (truncated, read the file for the rest)"
    return "\n\nProject memory (%s):\n%s" % (path, text)


def render_tools(tools, dialect="spark"):
    """Render tool specs the way the model's chat template would."""
    specs = [t.spec() for t in tools]
    if dialect == "qwen":
        specs = [{"type": "function", "function": spec} for spec in specs]
    return "\n".join(json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                     for spec in specs)


def dialect_for(model):
    """The tool-call format a model was trained on, guessed from its name."""
    return "qwen" if "qwen" in (model or "").lower() else "spark"


def build_system(tools, cwd, os_name="Windows 11", shell="cmd.exe", secret_names=(),
                 dialect="spark"):
    secrets = ""
    if secret_names:
        secrets = ("\nSecrets available as environment variables, referenced as %NAME% in sh. "
                   "Their values are never shown to you and must never be typed out: "
                   + ", ".join(sorted(secret_names)))
    block = QWEN_TOOLS if dialect == "qwen" else SPARK_TOOLS
    return SYSTEM_TEMPLATE.format(tools_block=block.format(tools=render_tools(tools, dialect)),
                                  cwd=cwd, os_name=os_name, shell=shell, secrets=secrets,
                                  state_dir=paths.project_dir(cwd),
                                  memory=_load_memory(cwd))


def strip_think(text):
    """Drop reasoning. The model's template opens generation inside <think>, so
    a reply often carries a closing tag with no opening one."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return THINK_RE.sub("", text).strip()


def extract_think(text):
    """The reasoning strip_think drops: everything before the last </think>
    (the opening tag is often implied) plus any complete <think> blocks."""
    parts = []
    if "</think>" in text:
        head, text = text.rsplit("</think>", 1)
        parts.append(head.replace("<think>", ""))
    parts.extend(m[len("<think>"):-len("</think>")] for m in THINK_RE.findall(text))
    return "\n".join(p.strip() for p in parts if p.strip())


def parse_tool_call(text):
    """Return (name, args) or None.

    Handles the native Spark format, Qwen's <function=...> XML, Hermes JSON,
    and an unterminated call (generation stopped on the stop sequence).
    """
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    body = match.group(1)
    function = FUNCTION_RE.search(body)
    if function:
        pairs = [(key, _param_value(value))
                 for key, value in PARAM_RE.findall(body, function.end())]
        return function.group(1).strip().strip('"'), _merge(pairs)
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


def _param_value(value):
    """qwen3_coder puts a value on lines of its own; drop exactly that one line
    break on each side (as vLLM's parser does), so indentation and blank lines
    inside a file's content survive."""
    value = value.split("</function>", 1)[0]
    for newline in ("\r\n", "\n"):
        if value.startswith(newline):
            value = value[len(newline):]
            break
    for newline in ("\r\n", "\n"):
        if value.endswith(newline):
            value = value[:-len(newline)]
            break
    return value


def _merge(pairs):
    """A repeated key is several values, not the last one. The model expresses
    "search for these four things" as four <arg_key>query</arg_key> pairs, and
    collapsing them into a dict silently threw three of them away."""
    args = {}
    for key, value in pairs:
        key = key.strip()
        args[key] = args[key] + "\n" + value if key in args else value
    return args


MARKUP = ("<arg_key>", "<arg_value>", "<tool_call>", "<function=", "<parameter=")


def malformed(args):
    """Tool-call markup left inside an argument value means the call was nested
    or otherwise broken. Executing it runs a fragment: one such call reached the
    shell as a command and came back as "< was unexpected at this time.", which
    tells the model nothing about what it did wrong."""
    return any(marker in str(value) for value in (args or {}).values()
               for marker in MARKUP)


def format_tool_call(name, args, dialect="spark"):
    """Inverse of parse_tool_call - useful for tests and replaying history."""
    if dialect == "qwen":
        parts = ["<tool_call>\n<function=%s>\n" % name]
        for key, value in args.items():
            parts.append("<parameter=%s>\n%s\n</parameter>\n" % (key, value))
        parts.append("</function>\n</tool_call>")
        return "".join(parts)
    parts = ["<tool_call>", name]
    for key, value in args.items():
        parts.append("<arg_key>%s</arg_key><arg_value>%s</arg_value>" % (key, value))
    parts.append("</tool_call>")
    return "".join(parts)


def example_call(dialect="spark"):
    """One well-formed call, quoted back to the model when it breaks the format."""
    if dialect == "qwen":
        return ("<tool_call><function=NAME><parameter=key>value</parameter></function>"
                "</tool_call>")
    return "<tool_call>NAME<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>"


def cut_after_call(text):
    """Drop whatever follows the first complete call: a model that keeps writing
    after </tool_call> is inventing the tool's answer."""
    end = text.find("</tool_call>")
    return text if end < 0 else text[:end + len("</tool_call>")]


def wrap_tool_response(text):
    return "<tool_response>%s</tool_response>" % text
