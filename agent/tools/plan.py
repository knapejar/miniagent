# -*- coding: utf-8 -*-
"""The plan - the agent's pinned memory.

It is written into a message pinned right behind the task. Context cropping
never touches it, so it survives runs of millions of tokens. Without it the
model has no record after a crop of which parts already worked, and breaks them
with its next edit.
"""
from ..protocol import PLAN_MAX_CHARS


class PlanTool(object):
    name = "plan"
    description = ("Write your checklist. It is pinned and survives context cropping. "
                   "Record what WORKS and must not be touched, and what is left.")
    params = {"text": {"type": "string"}}
    required = ("text",)
    danger = False

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return False

    def summary(self, args):
        text = str(args.get("text", ""))
        first = text.strip().splitlines()[0] if text.strip() else "(empty)"
        return "%s  (%d chars)" % (first[:60], len(text))

    def run(self, ctx, text="", **_):
        text = text[:PLAN_MAX_CHARS]
        ctx.set_plan(text)
        return "plan updated (%d chars)" % len(text)
