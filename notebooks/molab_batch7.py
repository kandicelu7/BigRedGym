# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Hardware policies on MoLab: train, evaluate, and download the runs and the
results. BATCH 9 (default) trains the 100 Hz curriculum tasks from scratch
(go2osc_{baseline,track,even}_curriculum_100hz: pushes ramp in over the first
500 iterations, start/stop over 800-1200, 95% torque clip, speed taper and a
joint-speed penalty above 85%; 2500 iterations). BATCH 8 trains
go2osc_baseline and tripedal_pronk in one stage (95% motor limits, 500 Hz
physics, 2000 iterations from scratch); BATCH 7 is the earlier two-stage
version described below.

Policies (go2osc_config.FOLLOWUP_BATCHES[7]), each fine-tuned from its
committed checkpoint at 95% of the actuator torque and speed limits with
500 Hz physics:
    deploy_robust_clip_ft2         from deploy_robust seed 2
    even2.5_deploy_track_clip_ft2  from even2.5_deploy_track seed 2 (candidate)

Evaluation: the grid, pushes and start/stop (friction 0.3 if RUN_FRICTION),
with a progress line per running run every minute.

Split over two notebooks with PART (BATCH 7/8 - 0: the first policy, 1: the
second; BATCH 9 - 0 and 1 take alternate runs of the queue, 8 and 7; None:
everything here). Run the cells top to bottom. Rerunning skips finished runs
and evaluations.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import subprocess
    import textwrap
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, mo, os, re, subprocess, textwrap, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Batch 7: fine-tune under actuator limits

    1. **Setup**: clone or update the repo and install the dependencies.
    2. **Settings**: pick `PART` (0 or 1 to split over two notebooks).
    3. **Train**: 5 seeds x 1000 iterations per policy.
    4. **Evaluate**: grid, pushes, start/stop (friction 0.3 optional), with
       progress printed every minute.
    5. **Download**: the trained runs and the results (two zips).
    """)
    return


@app.cell
def _(Path, os):
    WORK = Path(os.environ.get("QGYM_WORKDIR") or Path.home() / "qgym")
    REPO = WORK / "QGym"
    WARP_CACHE = WORK / "warp_cache"
    WARP_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ["PATH"] = f"{Path.home() / '.local/bin'}:{os.environ['PATH']}"
    os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE)
    os.environ["MPLBACKEND"] = "Agg"

    # MoLab's own venv variables would hijack the repo's uv commands; scrub them.
    _shadowing = (
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONEXECUTABLE",
    )
    ENV = {
        k: v
        for k, v in os.environ.items()
        if k not in _shadowing and not (k.startswith("UV_") and k != "UV_CACHE_DIR")
    }
    ENV["UV_PYTHON"] = "3.11"
    print(f"repo: {REPO}")
    return ENV, REPO, WORK


@app.cell
def _():
    BRANCH = "tripedal_test"
    REPO_URL = "https://github.com/kandicelu7/BigRedGym.git"
    return BRANCH, REPO_URL


@app.cell
def _(BRANCH, ENV, REPO, REPO_URL, subprocess, textwrap):
    _script = textwrap.dedent(f"""
        set -euo pipefail
        command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
        if [ -d {REPO}/.git ]; then
            git -C {REPO} fetch origin {BRANCH}
            git -C {REPO} checkout {BRANCH}
            git -C {REPO} pull --ff-only
        else
            git clone --branch {BRANCH} {REPO_URL} {REPO}
        fi
        cd {REPO}
        git log --oneline -1
        uv python install 3.11
        uv sync --frozen --extra gpu --no-dev
    """)
    subprocess.run(["bash", "-c", _script], check=True, env=ENV)
    print("\nsetup done")
    return


@app.cell
def _():
    # --- edit, then run --------------------------------------------------------
    # PART: 0 or 1 splits the two policies over two notebooks; None runs both.
    # BATCH 8: the hardware policies in one stage (tripedal_pronk,
    # go2osc_baseline; 2000 iterations from scratch). BATCH 7: the earlier
    # fine-tunes (1000 iterations from committed checkpoints).
    # BATCH 9: the 100 Hz curriculum tasks (baseline, speed tracking, even
    # load; 2500 iterations from scratch).
    BATCH = 9
    PART = None
    SEEDS = 5
    MAX_ITERATIONS = {7: 1000, 8: 2000, 9: 2500}[BATCH]
    DEVICE = "cuda:0"
    NUM_ENVS = 4096
    EVAL_JOBS = 2  # evaluations at once on the GPU
    # Friction-0.3 pass (readiness gate; ~60% of the grid's time): leave off
    # and run it later on the runs you pick, or set True to include it.
    RUN_FRICTION = False

    _all = {
        7: ["deploy_robust_clip_ft2", "even2.5_deploy_track_clip_ft2"],
        8: ["go2osc_baseline", "tripedal_pronk"],
        9: [
            "go2osc_baseline_curriculum_100hz",
            "go2osc_track_curriculum_100hz",
            "go2osc_even_curriculum_100hz",
        ],
    }[BATCH]
    # BATCH 9 splits the run queue (alternate runs) instead of the policies.
    SETTINGS = _all if PART is None or BATCH == 9 else [_all[PART]]
    # Hardware tasks (go2osc_baseline...) have their own names.
    TASKS = [s if s.startswith("go2osc_") else f"go2osc_wide_{s}" for s in SETTINGS]
    QUEUE = [(t, k) for k in range(SEEDS) for t in TASKS]
    if BATCH == 9 and PART is not None:
        QUEUE = QUEUE[PART::2]
    print(f"part {PART}: {SETTINGS}, {len(QUEUE)} runs of {MAX_ITERATIONS} iterations")
    for _t, _k in QUEUE:
        print(f"  {_t} seed{_k}")
    return (
        BATCH,
        DEVICE,
        EVAL_JOBS,
        MAX_ITERATIONS,
        NUM_ENVS,
        QUEUE,
        RUN_FRICTION,
        SETTINGS,
        TASKS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train

    About 40 min per run at 500 Hz physics (batches 7-8), 21 min at 100 Hz
    (batch 9, 2500 iterations). Time the first run before
    leaving it; rerun this cell after an interruption to continue.
    """)
    return


@app.cell
def _(DEVICE, ENV, MAX_ITERATIONS, NUM_ENVS, QUEUE, REPO, re, subprocess):
    import time as _time

    def _done(task, seed):
        return any(
            (REPO / "logs" / task).glob(f"*_seed{seed}/model_{MAX_ITERATIONS}.pt")
        )

    for _i, (_task, _seed) in enumerate(QUEUE, 1):
        if _done(_task, _seed):
            print(f"skip {_task} seed{_seed}: already trained", flush=True)
            continue
        print(f"\n=== [{_i}/{len(QUEUE)}] {_task} seed{_seed}", flush=True)
        _start = _time.monotonic()
        _cmd = [
            "uv", "run", "--frozen", "--extra", "gpu", "--no-dev", "scripts/train.py",
            "--task", _task, "--seed", str(_seed), "--run_name", f"seed{_seed}",
            "--device", DEVICE, "--headless", "--disable_wandb",
            "--num_envs", str(int(NUM_ENVS)),
            "--max_iterations", str(int(MAX_ITERATIONS)),
        ]  # fmt: skip
        _proc = subprocess.Popen(
            _cmd,
            cwd=REPO,
            env={**ENV, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for _line in _proc.stdout:
            _m = re.search(r"Iteration (\d+)/(\d+)", _line)
            if _m and int(_m.group(1)) % 100 == 0:
                print(f"  iteration {_m.group(1)}/{_m.group(2)}", flush=True)
            elif (
                "Initialized from" in _line or "Error" in _line or "Traceback" in _line
            ):
                print(_line, end="", flush=True)
        if _proc.wait() != 0:
            print(f"!!! {_task} seed{_seed} failed with code {_proc.returncode}")
        print(f"  {(_time.monotonic() - _start) / 60:.1f} min", flush=True)
    print("\ntraining finished")
    TRAIN_DONE = True
    return (TRAIN_DONE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate

    Every trained run on this machine is evaluated (evaluations already done
    are skipped): the speed x frequency grid, standardized pushes and the
    start/stop test, plus the friction-0.3 pass if `RUN_FRICTION`. Each
    script prints its queue, a progress line per running run every minute
    (conditions done, minutes in, estimated minutes left) and `done <run>` as
    each run finishes. Interrupting loses only the runs in progress.
    """)
    return


@app.cell
def _(DEVICE, ENV, EVAL_JOBS, REPO, RUN_FRICTION, SETTINGS, TRAIN_DONE, subprocess):
    assert TRAIN_DONE
    # The evaluation scripts import each other from scripts/; molab's Python
    # does not put a script's folder on the import path (safe-path mode).
    _env = {k: v for k, v in ENV.items() if k != "PYTHONSAFEPATH"}
    _env["PYTHONPATH"] = str(REPO / "scripts")
    _env["PYTHONUNBUFFERED"] = "1"
    _failed = []

    def _run(args):
        cmd = ["uv", "run", "--frozen", "--extra", "gpu", "--no-dev", *args]
        print("\n$ " + " ".join(args), flush=True)
        p = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=_env,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for line in p.stdout:
            if "Warning" not in line:
                print(line, end="", flush=True)
        if p.wait():
            print(f"!!! failed with code {p.returncode}", flush=True)
            _failed.append(args[0])

    _gpu = ["--device", DEVICE, "--jobs", str(EVAL_JOBS)]
    _run(["scripts/study_eval.py", "--wide", *_gpu])
    _run(["scripts/push_eval.py", "--study", "--wide", *_gpu])
    _run(["scripts/startstop_eval.py", "--study", "--settings", *SETTINGS, *_gpu])
    if RUN_FRICTION:
        _short = ["--robots", "16", "--settle", "6", "--measure", "6"]
        _run(["scripts/study_eval.py", "--wide", *_gpu, *_short, "--friction", "0.3",
              "--results", "friction0.3_wide"])  # fmt: skip
    if _failed:
        raise RuntimeError(f"failed: {_failed} (see above)")
    print("\nevaluation finished")
    EVAL_DONE = True
    return (EVAL_DONE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Download

    Two zips: the trained runs (final checkpoint, saved configs and code,
    paths `logs/<task>/<run>/...`) and the evaluation results (paths
    `logs/study_results/...` and `results/study/runs/...`). Unzip both into
    the repo on your PC.
    """)
    return


@app.cell
def _(BATCH, EVAL_DONE, MAX_ITERATIONS, PART, REPO, TASKS, WORK, mo, zipfile):
    assert EVAL_DONE
    _tag = "both" if PART is None else f"part{PART}"
    _out = WORK / "exports"
    _out.mkdir(parents=True, exist_ok=True)

    _runs_zip = _out / f"batch{BATCH}_runs_{_tag}.zip"
    with zipfile.ZipFile(_runs_zip, "w", compression=zipfile.ZIP_STORED) as _zf:
        for _task in TASKS:
            for _run in sorted((REPO / "logs" / _task).glob("*_seed*")):
                for _f in sorted(_run.rglob("*")):
                    _keep = not _f.name.startswith("model_") or (
                        _f.name == f"model_{MAX_ITERATIONS}.pt"
                    )
                    if _f.is_file() and _keep:
                        _zf.write(_f, _f.relative_to(REPO))
                print(f"runs: {_task}/{_run.name}")

    _res_zip = _out / f"batch{BATCH}_results_{_tag}.zip"
    _n = 0
    with zipfile.ZipFile(_res_zip, "w", compression=zipfile.ZIP_DEFLATED) as _zf:
        _roots = [REPO / "logs" / "study_results", REPO / "results" / "study" / "runs"]
        for _root in _roots:
            for _f in sorted(_root.rglob("*")):
                if _f.is_file() and any(f"{t}__" in _f.name for t in TASKS):
                    _zf.write(_f, _f.relative_to(REPO))
                    _n += 1
    print(f"\n{_runs_zip.name}: {_runs_zip.stat().st_size / 1e6:.1f} MB")
    print(f"{_res_zip.name}: {_n} files, {_res_zip.stat().st_size / 1e6:.1f} MB")
    mo.vstack(
        [
            mo.download(
                data=lambda p=_runs_zip: p.read_bytes(),
                filename=_runs_zip.name,
                label="Download trained runs",
            ),
            mo.download(
                data=lambda p=_res_zip: p.read_bytes(),
                filename=_res_zip.name,
                label="Download evaluation results",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
