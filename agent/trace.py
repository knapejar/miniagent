# -*- coding: utf-8 -*-
"""JSONL run trace - one line per event.

The trace is what makes two models comparable on the same task: how many steps,
how many tokens, where the context was cropped, where the model started looping.
"""
import json
import os
import time

from . import paths


class Trace(object):
    def __init__(self, directory=None, name=None):
        directory = paths.runs_dir() if directory is None else directory
        self.path = None
        self._fh = None
        if directory:
            os.makedirs(directory, exist_ok=True)
            name = name or time.strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(directory, "%s.jsonl" % name)
            self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, event, **fields):
        if not self._fh:
            return
        fields["ev"] = event
        fields["t"] = round(time.time(), 3)
        self._fh.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
