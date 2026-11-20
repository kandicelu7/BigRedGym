"""Run evaluation jobs a few at a time and print their progress while waiting.

The study evaluations (study_eval.py, push_eval.py, startstop_eval.py) each
start one subprocess per run, writing to a log. Every evaluated condition
logs a line like "<task> v<speed> w<omega>: ...", so counting those lines
gives each running job's progress. While jobs run, a line per job is printed
every `interval` seconds:

    [ 42 min] go2osc_wide_x__run.npz: 23/63 conditions (37%), 40 min in, ~68 min left

A job's slot is freed as soon as it finishes (not in start order).
"""

import os
import re
import time

CONDITION = re.compile(r" v[\d.]+ w[\d.]+: ")


def unbuffered(env):
    """Subprocess environment whose prints reach the log at once."""
    return {**env, "PYTHONUNBUFFERED": "1"}


class JobPool:
    def __init__(self, jobs, total, interval=60.0):
        self.jobs = max(1, jobs)
        self.total = total  # conditions per run
        self.interval = interval
        self.running = []  # (proc, name, log path, start time)
        self.failed = []
        self.start = time.monotonic()
        self._last = self.start

    def add(self, proc, name, log_path):
        self.running.append((proc, name, str(log_path), time.monotonic()))
        while len(self.running) >= self.jobs:
            self._wait()

    def close(self):
        while self.running:
            self._wait()

    def _wait(self):
        """Until at least one job finishes, printing progress meanwhile."""
        while True:
            done = [job for job in self.running if job[0].poll() is not None]
            if done:
                for job in done:
                    self.running.remove(job)
                    self._finish(job)
                return
            now = time.monotonic()
            if now - self._last >= self.interval:
                self._last = now
                for job in self.running:
                    print(self._progress(job, now), flush=True)
            time.sleep(2.0)

    def _finish(self, job):
        proc, name, _, started = job
        minutes = (time.monotonic() - started) / 60
        ok = proc.returncode == 0
        if not ok:
            self.failed.append(name)
        print(f"{'done  ' if ok else 'FAILED'}  {name} ({minutes:.0f} min)", flush=True)

    def _progress(self, job, now):
        _, name, log_path, started = job
        try:
            with open(log_path, errors="replace") as f:
                n = sum(1 for line in f if CONDITION.search(line))
        except OSError:
            n = 0
        ran = (now - started) / 60
        line = (
            f"[{(now - self.start) / 60:4.0f} min] {name}: {n}/{self.total} conditions"
        )
        if self.total:
            line += f" ({n / self.total:.0%}), {ran:.0f} min in"
        if n:
            line += f", ~{ran * (self.total - n) / n:.0f} min left"
        return line


def interval_from_env(default=60.0):
    """EVAL_PROGRESS_INTERVAL [s] overrides how often progress is printed."""
    return float(os.environ.get("EVAL_PROGRESS_INTERVAL", default))
