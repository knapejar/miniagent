# -*- coding: utf-8 -*-
"""The agent loop.

The loop is a generator of events and prints nothing itself. That keeps the UI
swappable (terminal, log, tests) without touching the logic, and makes the loop
straightforward to test.

Events: step, delta, usage, tool_call, tool_result, plan, crop, note, final,
        memory_save, end.
"""
import json
import time

from . import paths
from .llm import LLMError
from .guards import FailureMemory, Progress, ResultRepeat, failed
from .hints import Hints
from .jobs import JOB_WAIT
from .metrics import Metrics
from .protocol import example_call, extract_think, malformed, parse_tool_call, strip_think
from .secrets import redact
from .tools.shell import auto_background

OBS_MAX_CHARS = 3000
MAX_EMPTY_REPLIES = 3
OUTAGE_POLL = 30          # seconds between retries while the model server is down
LOOP_WINDOW = 3            # identical calls in a row that count as a stall
PLAN_NUDGE_AFTER = 15      # steps without a plan update before nudging
PLAN_NUDGE = ("\n\n[PLAN] You have not updated the plan for %d steps. "
              "Call the plan tool now: record what already works and must not be "
              "touched, and what is left.")

# The memory pass: one short run after the answer, whose only job is to write
# down what the next run would want to know. It is bounded and it can only
# touch files - a plan or an ask here would have nothing to act on.
MEMORY_MAX_STEPS = 4
MEMORY_MAX_CHARS = 4000
MEMORY_TOOLS = ("read", "write", "edit", "grep")
MEMORY_PROMPT = """[MEMORY] This is not a new task and it is not the old one.
The answer is already delivered - do not answer it again. Your one job now is to
write this project's memory file, and a tool call is the only way to do it.

The file is %s and this is what it holds right now:
---
%s
---
Call write on that exact path with the whole file: the lines above that are
still true, plus anything from this run worth having next time - a fact that was
looked up and may be wanted again, a host, a path, a login, a command that
works, a decision, a dead end not worth retrying. One line per note. Do not
write down what you did this run, and do not restate your answer as prose.

Keep it under %d characters; when a topic outgrows that, put it in its own file
next to the index and leave one line pointing at it. Tools: %s - nothing else
exists here. Reply with the single word NOTHING only when every line you would
write is already in the file above."""

MEMORY_NUDGE = """[MEMORY] That was prose, not a memory update, and nothing was
saved. Call write on %s now with the full contents of the file, or reply with
exactly NOTHING if there is truly nothing to add."""


def clip(text, limit=OBS_MAX_CHARS):
    """Trim tool output: keep head and tail, drop the middle."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return (text[:head] + "\n...[%d chars cut]...\n" % (len(text) - limit)
            + text[-(limit - head):])


class AgentLoop(object):
    def __init__(self, cfg, session, client, tools, trace=None, approve_fn=None,
                 cancel=None):
        self.cfg = cfg
        self.session = session
        self.client = client
        self.tools = {t.name: t for t in tools}
        self.trace = trace
        self.approve_fn = approve_fn
        self.cancel = cancel                 # threading.Event, set by the ESC watcher
        self.metrics = Metrics()
        self.failures = getattr(session, "failures", None) or FailureMemory()
        session.failures = self.failures
        session.cancel = cancel              # lets a waiting sh command notice esc
        self.repeats = ResultRepeat()
        self.progress = Progress()
        self.hints = Hints()
        self._since_plan = 0
        self._recent = []
        self._empty = 0
        self._pending_crop = 0

    # ------------------------------------------------------------- helpers
    def _log(self, event, **fields):
        if self.trace:
            self.trace.write(event, **fields)

    def _cancelled(self):
        return self.cancel is not None and self.cancel.is_set()

    def _call_model(self, on_delta):
        """Call the model; on context overflow, crop and retry once."""
        if self.session.should_crop():
            dropped = self.session.crop()
            if dropped:
                self.metrics.crops += 1
                self._log("crop", chars=dropped)
                self._pending_crop = dropped
        try:
            text, usage = self.client.chat(self.session.messages, on_delta, self.cancel)
        except LLMError as e:
            if e.overflow and self.session.crop():
                self.metrics.crops += 1
                self._log("crop_on_overflow", err=str(e)[:200])
                text, usage = self.client.chat(self.session.messages, on_delta, self.cancel)
            elif self.session.tool_role == "tool" and "tool" in str(e).lower():
                self.session.downgrade_tool_role()
                self._log("tool_role_fallback", err=str(e)[:200])
                text, usage = self.client.chat(self.session.messages, on_delta, self.cancel)
            else:
                raise
        self.session.calibrate(usage.prompt_tokens)
        self.metrics.add_usage(usage)
        return text, usage

    def _run_tool(self, name, args, key):
        tool = self.tools.get(name)
        if tool is None:
            return "error: unknown tool '%s'. Available: %s" % (
                name, ", ".join(sorted(self.tools))), 0.0
        if self.failures.blocked(key):
            self._log("blocked", tool=name, key=key[:200])
            return self.failures.message(key), 0.0
        if self._needs_approval(tool, args):
            if not self.approve_fn or not self.approve_fn(name, args):
                return "error: command rejected by the operator. Try a safer approach.", 0.0
        started = time.time()
        try:
            out = tool.run(self.session, **args)
        except TypeError as e:
            out = "error: bad arguments (%s)" % e
        except Exception as e:                       # a tool must never kill the loop
            out = "error: %s: %s" % (type(e).__name__, e)
        seconds = time.time() - started
        self.metrics.add_tool(name, seconds)
        return out, seconds

    def _needs_approval(self, tool, args):
        if self.cfg.approve == "none":
            return False
        if self.cfg.approve == "all":
            return getattr(tool, "danger", False)
        return getattr(tool, "needs_approval", lambda a: False)(args)

    @staticmethod
    def _key(name, args):
        return "%s|%s" % (name, json.dumps(args, sort_keys=True, default=str)[:300])

    def _deliver_jobs(self, step, wait=0.0):
        """Hand finished background work to the model as its own observation."""
        runner = getattr(self.session, "jobs", None)
        if runner is None or not runner.busy():
            return ""
        finished = runner.drain(wait)
        if not finished:
            return ""
        blocks = ["[BACKGROUND RESULT %s] %s (%.1fs)\n%s"
                  % (job, label, seconds, result)
                  for job, label, result, seconds in finished]
        for block in blocks:
            self.session.add_observation(clip(block))
            self._log("job_done", step=step, job=block[:120])
        return "%d background result(s) delivered" % len(finished)

    def _loop_warning(self, signature):
        self._recent.append(signature)
        if len(self._recent) >= LOOP_WINDOW and len(set(self._recent[-LOOP_WINDOW:])) == 1:
            self.metrics.loops += 1
            self._log("loop_detected", sig=signature[:200])
            return ("\n\n[WARNING] You have made the same call %d times in a row. It is not "
                    "working - change the approach, inspect the actual state with a different "
                    "tool, or move to the next subgoal." % LOOP_WINDOW)
        return ""

    def memory_pass(self, cancel=None):
        """One short bounded run after the final answer: the agent writes down
        what the next run in this project would want to know.

        It is driven by the caller, after the loop has ended, so it can run in a
        thread while the prompt is already up and the user is typing. `cancel`
        is its own event - the run's own esc watcher is gone by then.

        It runs on the same message list, so the prompt prefix in front of it is
        unchanged and the server reuses its KV cache instead of prefilling the
        whole conversation again. The messages it adds are dropped afterwards -
        the memory chatter would otherwise ride along in every later task.
        Nothing in here may break the run: the answer is already delivered.
        """
        if not getattr(self.cfg, "memory", True):
            return
        run_cancel, self.cancel = self.cancel, cancel   # the pass cancels on its own
        if self._cancelled():
            self.cancel = run_cancel
            return
        saved = list(self.session.messages)
        index = paths.memory_index(self.session.cwd)
        wrote, calls, reply, steps = [], [], "", 0
        yield "memory_start", {"index": index}
        try:
            # The index goes in verbatim. Making the model read it first cost a
            # round trip, and on the first run the read failed on a file that is
            # simply not there yet, which reads like a bug in the transcript.
            try:
                with open(index, encoding="utf-8") as fh:
                    current = fh.read().strip() or "(empty)"
            except OSError:
                current = "(does not exist yet - write creates it)"
            self.session.add_user(
                MEMORY_PROMPT % (index, current, MEMORY_MAX_CHARS, ", ".join(MEMORY_TOOLS)))
            nudged = False
            for steps in range(1, MEMORY_MAX_STEPS + 1):
                if self._cancelled():
                    break
                text, _ = self._call_model(None)     # not streamed - the answer is done
                reply = strip_think(text)
                self.session.add_assistant(reply, extract_think(text))
                call = parse_tool_call(text)
                if call is None:
                    # The model finished a task one message ago, and the system
                    # prompt tells it to end with plain prose; left alone it does
                    # that here too and nothing is saved. Ask once, plainly.
                    if wrote or nudged or reply.strip().upper().startswith("NOTHING"):
                        break
                    nudged = True
                    self.session.add_user(MEMORY_NUDGE % index)
                    continue
                name, args = call
                if name not in MEMORY_TOOLS:
                    out = ("error: %s is not available in the memory pass. Use %s."
                           % (name, ", ".join(MEMORY_TOOLS)))
                else:
                    out, _seconds = self._run_tool(name, args, self._key(name, args))
                    if name in ("write", "edit") and not out.startswith("error:"):
                        wrote.append(args.get("path") or index)
                observation = clip(redact(out, self.cfg.redact))
                # The pass reports itself as one collapsed line, not as steps of
                # its own - the task is already answered and this is bookkeeping.
                calls.append({"name": name, "text": observation[:200],
                              "error": observation.startswith("error:")})
                self.session.add_observation(observation)
        except (LLMError, OSError) as e:
            self._log("memory_failed", err="%s: %s" % (type(e).__name__, e))
            yield "note", {"level": "warn", "text": "memory pass failed: %s" % str(e)[:120]}
            return
        finally:
            # Restore the history whatever happened - _call_model may also have
            # cropped it, so put the list back rather than trimming an index.
            self.session.messages[:] = saved
            stopped, self.cancel = self._cancelled(), run_cancel
        if stopped:
            self._log("memory_cancelled", index=index, wrote=wrote, steps=steps)
            yield "memory_cancelled", {"index": index, "wrote": wrote}
            return
        self._log("memory", index=index, wrote=wrote, steps=steps, calls=calls,
                  reply=reply[:300])
        yield "memory_save", {"index": index, "wrote": wrote, "steps": steps,
                              "calls": calls, "text": reply}

    # ----------------------------------------------------------------- run
    def run(self, task, on_delta=None):
        """`on_delta(kind, text)` receives reply chunks as they arrive - the
        generator cannot yield them, because they would only be sampled after
        generation finished and the live output and tokens/s would stop being live."""
        self.session.start(task)
        self._recent, self._empty, self._pending_crop = [], 0, 0
        self._since_plan = 0
        self.hints.reset()
        self._log("task", task=task, model=self.cfg.model, config=self.cfg.as_dict())

        for step in range(1, self.cfg.max_steps + 1):
            self.metrics.steps = step
            yield "step", {"n": step, "max": self.cfg.max_steps}

            if self._cancelled():
                self._log("cancelled", step=step)
                yield "end", {"reason": "cancelled", "metrics": self.metrics.snapshot()}
                return

            waited = 0
            while True:
                try:
                    raw, usage = self._call_model(on_delta)
                    break
                except LLMError as e:
                    limit = getattr(self.cfg, "outage_wait", 0) or 0
                    if not e.transient or waited >= limit or self._cancelled():
                        self._log("error", err=str(e))
                        yield "end", {"reason": "error", "error": str(e),
                                      "metrics": self.metrics.snapshot()}
                        return
                    # the server is down or restarting: wait it out, keep the conversation
                    self._log("outage", err=str(e)[:300], waited=waited)
                    yield "note", {"level": "warn", "text": "model server unavailable (%s) - "
                                   "retrying in %d s, waited %d/%d min (esc to stop)" % (
                                       str(e)[:160], OUTAGE_POLL, waited // 60, limit // 60)}
                    if self.cancel is not None:
                        self.cancel.wait(OUTAGE_POLL)
                    else:
                        time.sleep(OUTAGE_POLL)
                    waited += OUTAGE_POLL
                    yield "step", {"n": step, "max": self.cfg.max_steps}   # fresh output for the retry

            if self._pending_crop:
                yield "crop", {"chars": self._pending_crop, "n": self.metrics.crops}
                self._pending_crop = 0
            yield "usage", {"usage": usage}

            if self._cancelled():
                self._log("cancelled", step=step)
                yield "end", {"reason": "cancelled", "metrics": self.metrics.snapshot()}
                return

            body = strip_think(raw)
            call = parse_tool_call(body)
            reasoning = "\n".join(t for t in (getattr(usage, "reasoning", ""),
                                              extract_think(raw)) if t)
            self.session.add_assistant(body, reasoning)

            if call is None:
                # An empty reply or a cut-off at the token limit is NOT a finished
                # task - the model usually spent the whole budget on reasoning.
                if not body or usage.finish_reason == "length":
                    self._empty += 1
                    self.metrics.empty_replies += 1
                    self._log("empty_reply", step=step, finish=usage.finish_reason,
                              reasoning_tokens=usage.reasoning_tokens)
                    yield "note", {"level": "warn",
                                   "text": "no tool call (finish=%s, %d reasoning tokens) "
                                           "- correction %d/%d"
                                           % (usage.finish_reason, usage.reasoning_tokens,
                                              self._empty, MAX_EMPTY_REPLIES)}
                    if self._empty >= MAX_EMPTY_REPLIES:
                        self._log("stalled", step=step)
                        yield "end", {"reason": "stalled", "metrics": self.metrics.snapshot()}
                        return
                    correction = ("[ERROR] Your last reply contained no <tool_call> and was "
                                  "cut off at the token limit. Keep your reasoning short, then "
                                  "emit exactly ONE <tool_call> now to make progress.")
                    self.session.add_observation(
                        correction + self.session.reminder(step, self.cfg.max_steps))
                    continue
                for text in auto_background(self.session):
                    yield "note", {"level": "info", "text": "undecided command: " + text[:140]}
                delivered = self._deliver_jobs(step, wait=JOB_WAIT)
                if delivered:
                    # Work it asked for is only now arriving; answering without
                    # it would throw away what it waited for.
                    yield "note", {"level": "info", "text": delivered}
                    continue
                self._log("final", step=step, text=body, metrics=self.metrics.snapshot())
                yield "final", {"text": body}
                # The memory pass is NOT run here. It would hold the keyboard and
                # the screen for a couple of seconds after the answer, and the
                # prompt is what the user wants at that moment; the caller runs
                # `memory_pass()` in the background instead.
                yield "end", {"reason": "final", "metrics": self.metrics.snapshot(),
                              "memory_due": bool(getattr(self.cfg, "memory", True))}
                return

            if usage.finish_reason == "length":
                # Cut off mid-call: the arguments are incomplete, so running it
                # would act on a fragment - which is how a write once truncated a
                # file to zero bytes.
                self._log("truncated_call", step=step, tool=call[0])
                yield "note", {"level": "warn",
                               "text": "tool call cut off at the token limit - not run"}
                self.session.add_observation(
                    "[ERROR] Your tool call was cut off at the token limit, so it was NOT "
                    "run. Make the arguments smaller: write a file in several parts, or "
                    "shorten the content." + self.session.reminder(step, self.cfg.max_steps))
                continue

            self._empty = 0
            name, args = call
            if malformed(args):
                self._log("malformed_call", step=step, tool=name)
                yield "note", {"level": "warn",
                               "text": "malformed tool call (nested markup) - not run"}
                self.session.add_observation(
                    "[ERROR] That call was malformed and was NOT run: an argument value "
                    "still contained tool-call markup. One call looks like "
                    + example_call(getattr(self.cfg, "dialect", "spark"))
                    + " and its values are plain text."
                    + self.session.reminder(step, self.cfg.max_steps))
                continue
            if name != "wait":
                for text in auto_background(self.session):
                    self._log("auto_background", step=step, text=text[:200])
                    yield "note", {"level": "info", "text": "undecided command: " + text[:140]}
            yield "tool_call", {"name": name, "args": args, "tool": self.tools.get(name)}

            key = self._key(name, args)
            out, seconds = self._run_tool(name, args, key)
            observation = clip(redact(out, self.cfg.redact))
            self.failures.record(key, failed(observation))
            yield "tool_result", {"name": name, "text": observation, "seconds": seconds,
                                  "error": observation.startswith("error:")}
            if name == "plan":
                self._since_plan = 0
                yield "plan", {"text": self.session.plan_text}
                self._log("plan", step=step, text=self.session.plan_text)
            else:
                self._since_plan += 1

            if name == "ask":
                self._log("asked", step=step, question=observation)
                yield "final", {"text": observation}
                yield "end", {"reason": "asked", "metrics": self.metrics.snapshot()}
                return

            warning = (self._loop_warning(key) or self.progress.push(name, args, observation)
                       or self.repeats.push(observation)
                       or self.hints.match(name, observation))

            self._log("step", step=step, tool=name, args=args, obs=observation[:2000],
                      pt=usage.prompt_tokens, ct=usage.completion_tokens,
                      tps=round(usage.tps, 1), cum=self.metrics.total_tokens,
                      warn=warning.strip()[:120] or None)
            if self._since_plan >= PLAN_NUDGE_AFTER:
                warning += PLAN_NUDGE % self._since_plan
                self._since_plan = 0
            if warning:
                yield "note", {"level": "warn", "text": warning.strip()[:160]}
            delivered = self._deliver_jobs(step)
            if delivered:
                yield "note", {"level": "info", "text": delivered}
            reminder = self.session.reminder(step, self.cfg.max_steps)
            if reminder:
                self._log("goal_reminded", step=step)
            self.session.add_observation(observation + warning + reminder)

        self._log("limit", steps=self.cfg.max_steps, metrics=self.metrics.snapshot())
        yield "end", {"reason": "limit", "metrics": self.metrics.snapshot()}
