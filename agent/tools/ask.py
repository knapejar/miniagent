# -*- coding: utf-8 -*-
"""Stop and ask instead of retrying something that cannot work."""


class AskTool(object):
    name = "ask"
    description = ("Stop and ask the operator when you are blocked by something you cannot "
                   "do yourself, instead of retrying a command that already failed.")
    params = {"question": {"type": "string"}}
    required = ("question",)
    danger = False

    def spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return False

    def summary(self, args):
        return str(args.get("question", ""))[:80]

    def run(self, ctx, question="", **_):
        return question
