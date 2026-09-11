# -*- coding: utf-8 -*-
"""Run slow tool work in parallel, and optionally in the background.

Search is the slowest tool but not because of the network: four fresh searches
take 2.8s one after another and 1.0s at once. The real cost is that the model
spends 3-4s generating each call, so the win comes from asking for several things
in one call rather than from making one call return sooner.

Background jobs earn their keep in the other case: when the agent has something
useful to do while the answer is on its way.
"""
import concurrent.futures as futures
import itertools
import threading
import time

MAX_WORKERS = 6
JOB_WAIT = 20          # seconds the loop will wait for pending jobs before ending


def parallel(items, fn, workers=MAX_WORKERS):
    """Map fn over items concurrently, keeping the input order."""
    items = list(items)
    if len(items) <= 1:
        return [_safe(fn, item) for item in items]
    with futures.ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(lambda item: _safe(fn, item), items))


def _safe(fn, item):
    try:
        return fn(item)
    except Exception as e:
        return "error: %s: %s" % (type(e).__name__, e)


class JobRunner(object):
    """Fire-and-collect. Finished results are handed to the loop, which appends
    them to the next observation, so the model does not need a collect tool."""

    def __init__(self, workers=MAX_WORKERS):
        self.pool = futures.ThreadPoolExecutor(max_workers=workers)
        self.pending = {}
        self.counter = itertools.count(1)
        self.lock = threading.Lock()

    def submit(self, label, fn):
        job_id = "job%d" % next(self.counter)
        with self.lock:
            self.pending[job_id] = (label, self.pool.submit(fn), time.time())
        return job_id

    def busy(self):
        with self.lock:
            return len(self.pending)

    def drain(self, wait=0.0):
        """Return [(job_id, label, result, seconds)] for everything that has
        finished. With wait > 0, block for that long for the rest."""
        deadline = time.time() + wait
        done = []
        while True:
            with self.lock:
                items = list(self.pending.items())
            for job_id, (label, future, started) in items:
                if future.done():
                    with self.lock:
                        self.pending.pop(job_id, None)
                    done.append((job_id, label, _result(future), time.time() - started))
            if done or time.time() >= deadline or not self.busy():
                return done
            time.sleep(0.1)

    def shutdown(self):
        self.pool.shutdown(wait=False)


def _result(future):
    try:
        return future.result(timeout=0)
    except Exception as e:
        return "error: %s: %s" % (type(e).__name__, e)
