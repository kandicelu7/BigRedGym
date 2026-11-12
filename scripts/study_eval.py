"""Evaluate every trained reward-study run with grid_eval.py.

Finds runs under logs/go2osc_study_<setting>/<run>/ and evaluates each over the
(speed, frequency) grid, several at a time, writing
logs/study_results/<task>__<run>[__it<N>].png / .npz / .log. Runs already
evaluated are skipped, so it can be re-run as more runs arrive.

Usage:
    uv run --frozen scripts/study_eval.py                    # latest checkpoints
    uv run --frozen scripts/study_eval.py --checkpoint 1250  # convergence check
    uv run --frozen scripts/study_eval.py --robots 32        # faster, noisier
Then summarize with scripts/study_summary.py.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from gym.envs.go2.go2osc_config import STUDY_SETTINGS, study_task_name
from gym.utils.helpers import GYM_ROOT_DIR

RESULTS = Path(GYM_ROOT_DIR) / "logs" / "study_results"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate reward-study runs")
    parser.add_argument("--checkpoint", type=int, default=-1, help="default: latest")
    parser.add_argument("--robots", type=int, default=64, help="robots per cell")
    parser.add_argument("--jobs", type=int, default=4, help="evaluations at once")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per job")
    parser.add_argument("--dry_run", action="store_true", help="list, don't run")
    return parser.parse_args(argv)


def study_runs():
    """(task, run directory name) of every study run with checkpoints."""
    found = []
    for setting in STUDY_SETTINGS:
        task = study_task_name(setting)
        for run in sorted((Path(GYM_ROOT_DIR) / "logs" / task).glob("*")):
            if any(run.glob("model_*.pt")):
                found.append((task, run.name))
    return found


def out_path(task, run, checkpoint):
    suffix = "" if checkpoint == -1 else f"__it{checkpoint}"
    return RESULTS / f"{task}__{run}{suffix}.png"


def main():
    args = get_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = []
    for task, run in study_runs():
        out = out_path(task, run, args.checkpoint)
        if out.with_suffix(".npz").exists():
            print(f"done    {out.name}")
        else:
            todo.append((task, run, out))
            print(f"queued  {out.name}")
    print(f"{len(todo)} to evaluate")
    if args.dry_run or not todo:
        return

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    script = Path(__file__).with_name("grid_eval.py")
    running, failed = [], []

    def finish(entry):
        proc, name = entry
        if proc.wait() != 0:
            failed.append(name)
        print(f"{'FAILED' if proc.returncode else 'done  '}  {name}", flush=True)

    for task, run, out in todo:
        cmd = [
            sys.executable,
            str(script),
            "--task",
            task,
            "--load_run",
            run,
            "--checkpoint",
            str(args.checkpoint),
            "--robots",
            str(args.robots),
            "--out",
            str(out),
        ]
        log = open(out.with_suffix(".log"), "w")
        proc = subprocess.Popen(
            cmd, cwd=script.parent, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        running.append((proc, out.name))
        if len(running) >= args.jobs:
            finish(running.pop(0))
    for entry in running:
        finish(entry)
    if failed:
        raise SystemExit(f"failed (see .log next to each): {failed}")


if __name__ == "__main__":
    main()
