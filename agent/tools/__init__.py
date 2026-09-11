# -*- coding: utf-8 -*-
"""Tool registry.

There are deliberately few tools and they are orthogonal. Every line of a tool
description is paid for in EVERY turn of the model, so brevity here is not
cosmetics, it is throughput.
"""


class Tool(object):
    name = ""
    description = ""
    params = {}
    required = ()
    danger = False        # True = always confirm in approve=auto mode

    def spec(self):
        return {"name": self.name,
                "description": self.description,
                "parameters": {"type": "object",
                               "properties": self.params,
                               "required": list(self.required)}}

    def needs_approval(self, args):
        return self.danger

    def run(self, ctx, **args):
        raise NotImplementedError

    def summary(self, args):
        """Short one-line description of the call, for the UI."""
        return ", ".join("%s=%s" % (k, str(v).replace("\n", " ")[:60])
                         for k, v in args.items())


from .shell import ShellTool                       # noqa: E402
from .files import ReadTool, WriteTool, EditTool   # noqa: E402
from .search import GrepTool                       # noqa: E402
from .plan import PlanTool                         # noqa: E402
from .ask import AskTool                           # noqa: E402
from .web import WebSearchTool, BrowseTool         # noqa: E402


def default_tools():
    return [ShellTool(), ReadTool(), WriteTool(), EditTool(), GrepTool(),
            WebSearchTool(), BrowseTool(), PlanTool(), AskTool()]


def by_name(tools):
    return {t.name: t for t in tools}


__all__ = ["Tool", "ShellTool", "ReadTool", "WriteTool", "EditTool", "GrepTool",
           "PlanTool", "AskTool", "WebSearchTool", "BrowseTool",
           "default_tools", "by_name"]
