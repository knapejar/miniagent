# -*- coding: utf-8 -*-
"""MCP servers as tools - the client half of the Model Context Protocol.

`--mcp-config` takes the same JSON Claude Code does, inline or as a file:

    {"mcpServers": {
        "notes": {"command": "node", "args": ["notes-mcp.js"], "env": {"X": "1"}},
        "docs":  {"type": "http", "url": "https://example.com/mcp",
                  "headers": {"Authorization": "Bearer ..."}}
    }}

Every tool a server lists becomes a tool named mcp__<server>__<tool>, called
with the arguments the model gives it. stdio servers speak newline-delimited
JSON-RPC on their stdin/stdout; http servers take one JSON-RPC request per POST
and answer with JSON or a short event stream.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request

PROTOCOL = "2025-06-18"
START_TIMEOUT = 30          # seconds a server has to answer initialize and tools/list
CALL_TIMEOUT = 3600         # a tool may wait for a human; the run's cancel still stops it
DESCRIPTION_MAX = 1500


class MCPError(Exception):
    pass


def load_config(values):
    """{name: server} from every --mcp-config value: inline JSON or a path."""
    servers = {}
    for value in values or ():
        text = value.strip()
        if not text.startswith("{"):
            with open(text, encoding="utf-8-sig") as fh:
                text = fh.read()
        data = json.loads(text)
        servers.update(data.get("mcpServers", data))
    return servers


class StdioServer(object):
    def __init__(self, name, spec):
        self.name = name
        command = spec.get("command") or ""
        argv = [shutil.which(command) or command] + [str(a) for a in spec.get("args") or []]
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, cwd=spec.get("cwd") or None, bufsize=0)
        self.lock = threading.Lock()
        self.waiting = {}            # id -> [Event, response]
        self.next_id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                message = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            slot = self.waiting.get(message.get("id")) if isinstance(message, dict) else None
            if slot and ("result" in message or "error" in message):
                slot[1] = message
                slot[0].set()
        for slot in list(self.waiting.values()):    # the server is gone
            slot[0].set()

    def _write(self, message):
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def notify(self, method, params=None):
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method, params=None, timeout=START_TIMEOUT, cancel=None):
        with self.lock:
            self.next_id += 1
            rid = self.next_id
        slot = self.waiting[rid] = [threading.Event(), None]
        try:
            self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            deadline = time.time() + timeout
            while not slot[0].wait(0.2):
                if cancel is not None and cancel.is_set():
                    self.notify("notifications/cancelled", {"requestId": rid})
                    raise MCPError("cancelled")
                if time.time() > deadline:
                    raise MCPError("no answer in %ds" % timeout)
        except OSError as e:
            raise MCPError("server is gone (%s)" % e)
        finally:
            self.waiting.pop(rid, None)
        if slot[1] is None:
            raise MCPError("server exited")
        return _result(slot[1])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            pass
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass


class HttpServer(object):
    def __init__(self, name, spec):
        self.name = name
        self.url = spec["url"]
        self.headers = dict(spec.get("headers") or {})
        self.session = None
        self.next_id = 0

    def _post(self, message, timeout):
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL}
        headers.update(self.headers)
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = urllib.request.Request(self.url, json.dumps(message).encode("utf-8"),
                                         headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                self.session = response.headers.get("Mcp-Session-Id") or self.session
                body = response.read().decode("utf-8", "replace")
                kind = response.headers.get("Content-Type", "")
        except OSError as e:
            raise MCPError(str(e))
        if "id" not in message or not body.strip():
            return None
        if "event-stream" in kind:
            for line in body.splitlines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:])
                    except ValueError:
                        continue
                    if data.get("id") == message["id"]:
                        return data
            raise MCPError("no answer in the event stream")
        return json.loads(body)

    def notify(self, method, params=None):
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}}, START_TIMEOUT)

    def request(self, method, params=None, timeout=START_TIMEOUT, cancel=None):
        self.next_id += 1
        return _result(self._post({"jsonrpc": "2.0", "id": self.next_id, "method": method,
                                   "params": params or {}}, timeout))

    def close(self):
        pass


def _result(message):
    if "error" in message:
        error = message["error"] or {}
        raise MCPError(error.get("message") or str(error))
    return message.get("result") or {}


class MCPTool(object):
    """One tool of one server, shaped like the built-in tools."""
    danger = False

    def __init__(self, server, info):
        self.server = server
        self.remote_name = info["name"]
        self.name = "mcp__%s__%s" % (server.name, info["name"])
        self.description = (info.get("description") or "")[:DESCRIPTION_MAX]
        schema = dict(info.get("inputSchema") or {})
        schema["type"] = "object"
        schema.setdefault("properties", {})
        self.schema = schema
        self.params = schema["properties"]
        self.required = tuple(schema.get("required") or ())

    def spec(self):
        return {"name": self.name, "description": self.description, "parameters": self.schema}

    def needs_approval(self, args):
        return False

    def summary(self, args):
        return json.dumps(args, ensure_ascii=False)[:100]

    def _coerce(self, args):
        """The tool-call parsers hand every value over as a string; a server
        validating its schema wants the number, flag or list it declared."""
        out = {}
        for key, value in args.items():
            kind = (self.params.get(key) or {}).get("type")
            if isinstance(value, str) and kind in ("number", "integer", "boolean", "array", "object"):
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            out[key] = value
        return out

    def run(self, ctx, **args):
        try:
            result = self.server.request("tools/call",
                                         {"name": self.remote_name, "arguments": self._coerce(args)},
                                         timeout=CALL_TIMEOUT, cancel=getattr(ctx, "cancel", None))
        except MCPError as e:
            return "error: %s: %s" % (self.name, e)
        parts = []
        for block in result.get("content") or []:
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif block.get("type") == "resource":
                resource = block.get("resource") or {}
                parts.append(resource.get("text") or "[resource %s]" % resource.get("uri", ""))
            else:
                parts.append("[%s]" % block.get("type"))
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        text = "\n".join(parts).strip() or "(no output)"
        return ("error: " + text) if result.get("isError") else text


def start(servers, say=None):
    """Start every server in parallel. Returns (tools, statuses, handles)."""
    results = {}

    def one(name, spec):
        server = None
        try:
            kind = spec.get("type") or ("http" if spec.get("url") else "stdio")
            if kind in ("http", "streamable-http", "sse"):
                server = HttpServer(name, spec)
            else:
                server = StdioServer(name, spec)
            server.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                          "clientInfo": {"name": "miniagent", "version": "0.2.0"}})
            server.notify("notifications/initialized")
            listed = server.request("tools/list").get("tools") or []
            results[name] = (server, [MCPTool(server, t) for t in listed], "connected")
        except Exception as e:
            if server is not None:
                server.close()
            results[name] = (None, [], "failed")
            if say:
                say("mcp %s failed: %s" % (name, e))

    threads = [threading.Thread(target=one, args=item, daemon=True) for item in servers.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(START_TIMEOUT + 5)
    tools, statuses, handles = [], [], []
    for name in servers:
        server, found, status = results.get(name, (None, [], "failed"))
        tools.extend(found)
        statuses.append({"name": name, "status": status})
        if server is not None:
            handles.append(server)
    return tools, statuses, handles
