"""Evaluate every trained reward-study run with grid_eval.py.

Finds runs under logs/go2osc_study_<setting>/<run>/ and evaluates each over the
(speed, frequency) grid, several at a time. Results (.npz) and each
evaluation's log go to logs/study_results/runs/ (logs in runs/logs/), the
per-run heatmaps (.png) to results/study/runs/. Runs already evaluated are
skipped, so it can be re-run as more runs arrive.

Usage:
    uv run --frozen scripts/study_eval.py                    # latest checkpoints
    uv run --frozen scripts/study_eval.py --checkpoint 1250  # convergence check
    uv run --frozen scripts/study_eval.py --robots 32        # faster, noisier
    uv run --frozen scripts/study_eval.py --force            # redo, overwriting
    # Short pass at low friction, its own results folder (no heatmaps):
    uv run --frozen scripts/study_eval.py --wide --friction 0.3 --robots 16 \
        --settle 6 --measure 6 --results friction0.3_wide
Then summarize with scripts/study_summary.py.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from eval_progress import JobPool, interval_from_env, unbuffered
from gym.envs.go2.go2osc_config import (
    HARDWARE_TASKS,
    STUDY_SETTINGS,
    SWEEP_OMEGAS,
    SWEEP_SPEEDS,
    WIDE_OMEGAS,
    WIDE_SETTINGS,
    WIDE_SPEEDS,
    study_task_name,
)
from gym.utils.helpers import GYM_ROOT_DIR
from study_summary import DATA, FIGURES

RESULTS = DATA / "runs"  # .npz per run
LOGS = RESULTS / "logs"  # one log per evaluation
PLOTS = FIGURES / "runs"  # heatmaps per run


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate reward-study runs")
    parser.add_argument("--checkpoint", type=int, default=-1, help="default: latest")
    parser.add_argument("--robots", type=int, default=64, help="robots per cell")
    parser.add_argument("--jobs", type=int, default=4, help="evaluations at once")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per job")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda:0 (MuJoCo Warp, much faster)",
    )
    parser.add_argument(
        "--friction", type=float, default=None, help="fixed contact friction"
    )
    parser.add_argument("--settle", type=float, default=10.0, help="settling time [s]")
    parser.add_argument("--measure", type=float, default=10.0, help="measured time [s]")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="folder under logs/study_results/ for these results (default: runs); "
        "heatmaps then go next to the .npz",
    )
    parser.add_argument("--dry_run", action="store_true", help="list, don't run")
    parser.add_argument(
        "--wide",
        action="store_true",
        help="the wide-range study (go2osc_wide_*: to 4.5 m/s and 6 Hz)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-evaluate runs that already have results, overwriting them",
    )
    return parser.parse_args(argv)


def study_runs(wide=False):
    """(task, run directory name) of every study run with checkpoints."""
    found = []
    tasks = [study_task_name(s, wide) for s in (WIDE_SETTINGS if wide else STUDY_SETTINGS)]
    if wide:
        tasks += list(HARDWARE_TASKS)  # own names, evaluated like the wide study
    for task in tasks:
        for run in sorted((Path(GYM_ROOT_DIR) / "logs" / task).glob("*")):
            if any(run.glob("model_*.pt")):
                found.append((task, run.name))
    return found


def out_path(task, run, checkpoint, results=RESULTS):
    """Results file of one evaluation (its heatmap: same name in PLOTS)."""
    suffix = "" if checkpoint == -1 else f"__it{checkpoint}"
    return results / f"{task}__{run}{suffix}.npz"


def main():
    args = get_args()
    results = RESULTS if args.results is None else DATA / args.results
    logs = results / "logs"
    plots = PLOTS if args.results is None else results / "heatmaps"
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    todo = []
    for task, run in study_runs(args.wide):
        out = out_path(task, run, args.checkpoint, results)
        if out.exists() and not args.force:
            print(f"done    {out.name}")
        else:
            todo.append((task, run, out))
            print(f"queued  {out.name}")
    print(f"{len(todo)} to evaluate")
    if args.dry_run or not todo:
        return

    env = unbuffered(dict(os.environ, OMP_NUM_THREADS=str(args.threads)))
    script = Path(__file__).with_name("grid_eval.py")
    cells = (
        len(WIDE_SPEEDS) * len(WIDE_OMEGAS)
        if args.wide
        else len(SWEEP_SPEEDS) * len(SWEEP_OMEGAS)
    )
    pool = JobPool(args.jobs, cells, interval_from_env())

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
            str(plots / f"{out.stem}.png"),
            "--data_out",
            str(out),
            "--device",
            args.device,
            "--settle",
            str(args.settle),
            "--measure",
            str(args.measure),
        ]
        if args.friction is not None:
            cmd += ["--friction", str(args.friction)]
        log_path = logs / f"{out.stem}.log"
        proc = subprocess.Popen(
            cmd,
            cwd=script.parent,
            env=env,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        pool.add(proc, out.name, log_path)
    pool.close()
    if pool.failed:
        raise SystemExit(f"failed (see the .log of each in {logs}): {pool.failed}")


if __name__ == "__main__":
    main()
