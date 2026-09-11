# -*- coding: utf-8 -*-
"""Detectors that turn repeated failure into actionable feedback."""
import hashlib
import re

FAILED_EXIT = re.compile(r"^exit=(-?\d+)", re.M)

BLOCKED = ("error: this exact call already failed {n} times with the same result. "
           "It is blocked now. Read the last error again, or try a different approach.")
STUCK = ("\n\n[WARNING] The last %d calls brought back nothing you had not already "
         "seen. Repeating the same kind of request will not help - act on what you "
         "have, or change the target.")
SIMILAR = ("\n\n[WARNING] You have made %d near-identical calls to %s. Rephrasing "
           "the same request will not help. Change what you are doing.")
SAME_RESULT = ("\n\n[WARNING] The last {n} tool calls returned an identical result. Nothing is "
               "changing. Inspect the actual state with a different tool, or move on.")


def digest(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def failed(observation):
    if observation.startswith("error:"):
        return True
    match = FAILED_EXIT.search(observation)
    return bool(match) and match.group(1) != "0"


class FailureMemory(object):
    """Survives across turns: a command that keeps failing stops being run."""

    def __init__(self, limit=3):
        self.limit = limit
        self.counts = {}

    def record(self, key, did_fail):
        if did_fail:
            self.counts[key] = self.counts.get(key, 0) + 1
        else:
            self.counts.pop(key, None)

    def blocked(self, key):
        return self.counts.get(key, 0) >= self.limit

    def message(self, key):
        return BLOCKED.format(n=self.counts.get(key, 0))


class Progress(object):
    """Progress is novelty, not variety of tools.

    An earlier version warned whenever the same tool was called several times in
    a row. That is exactly what researching a site looks like - six browse calls
    on six different pages - and the warning cut a good run short. What actually
    distinguishes spinning from working is whether a call brings back anything
    that was not already known.
    """

    URL = re.compile(r"https?://[^\s\)\]>\"']+")

    def __init__(self, similar=3, unproductive=3, window=10):
        self.similar_limit, self.unproductive_limit = similar, unproductive
        self.window = window
        self.shapes = []
        self.seen_urls = set()
        self.seen_results = set()
        self.barren = 0

    @staticmethod
    def _shape(name, args):
        words = re.findall(r"[a-z0-9]+", str(args).lower())
        return name + "|" + " ".join(sorted(set(words))[:12])

    def push(self, name, args, observation=""):
        shape = self._shape(name, args)
        self.shapes.append(shape)
        del self.shapes[:-self.window]
        if self.shapes.count(shape) >= self.similar_limit:
            self.shapes = []
            return SIMILAR % (self.similar_limit, name)

        urls = set(self.URL.findall(observation or ""))
        fingerprint = digest(observation or "")
        fresh = bool(urls - self.seen_urls) or fingerprint not in self.seen_results
        self.seen_urls |= urls
        self.seen_results.add(fingerprint)

        self.barren = 0 if fresh else self.barren + 1
        if self.barren >= self.unproductive_limit:
            self.barren = 0
            return STUCK % self.unproductive_limit
        return ""


class ResultRepeat(object):
    """Catches thrashing that the call-based detector misses: the model varies
    the command but keeps getting the same output."""

    def __init__(self, window=3):
        self.window = window
        self.recent = []

    def push(self, observation):
        self.recent.append(digest(observation))
        del self.recent[:-self.window]
        if len(self.recent) == self.window and len(set(self.recent)) == 1:
            self.recent = []
            return SAME_RESULT.format(n=self.window)
        return ""
