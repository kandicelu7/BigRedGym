# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
#
# [tool.marimo.runtime]
# # Run this like a normal notebook: top to bottom, one cell at a time. "lazy"
# # turns marimo's reactive execution off, so editing a parameter only marks the
# # cells below it stale instead of instantly re-firing them -- which matters a
# # lot when the cell below clones a repo or launches a training run.
# on_cell_change = "lazy"
# ///
"""QGym cloud training as a marimo notebook (runs on MoLab).

The marimo port of ``notebooks/colab_train.ipynb``: clone/refresh the repo,
install the locked GPU dependency graph with uv, train, then download the
checkpoints as a zip.

    uv run marimo edit notebooks/molab_train.py     # local
    # MoLab: upload this file to molab.marimo.io, pick a GPU runtime, run.

Only marimo itself is imported here; the repo and its dependencies live in a
separate uv environment driven through subprocess, exactly as in the Colab
notebook. Disconnect the runtime when you are done or it keeps burning compute.

How to use it: run the cells in order. Parameters live in their own cell above
each step -- edit the constants, run that cell, then run the step below it. A
step that fails raises, so a red cell means stop and read the log rather than
carry on. Nothing runs on its own, and nothing re-runs behind your back.
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
    # BigRedGym Training in the Cloud!

    | Section | Does |
    |---|---|
    | 1 Setup | Clones/Updates the repo and installs the locked GPU deps with `uv` |
    | 2 Train | Runs `scripts/train.py`, streaming the log below the cell |
    | 3 Download | Zips a run's checkpoints and hands you the download |

    For each cell below, run it using **Shift + Enter**, or hit the small yellow play button on the top right.
    """)
    return


@app.cell
def _(Path, os):
    # --- environment ---------------------------------------------------------
    WORK = Path(os.environ.get("QGYM_WORKDIR") or Path.home() / "qgym")
    REPO = WORK / "QGym"
    WARP_CACHE = WORK / "warp_cache"
    WARP_CACHE.mkdir(parents=True, exist_ok=True)

    # uv installs to ~/.local/bin; keep the compiled warp kernels across runs.
    os.environ["PATH"] = f"{Path.home() / '.local/bin'}:{os.environ['PATH']}"
    os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE)
    os.environ["MPLBACKEND"] = "Agg"

    # MoLab runs this notebook inside its own uv venv on Python 3.13 and exports
    # VIRTUAL_ENV / UV_* / PYTHON*. Inherited, those hijack interpreter selection
    # for the repo ("resolved to Python 3.13.11, incompatible with ==3.11.*"), so
    # the repo's uv commands get a scrubbed environment pinned to 3.11.
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

    print(f"work dir : {WORK}")
    print(f"repo     : {REPO}")
    return ENV, REPO, WORK


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 Setup
    
    Here we clone the repository from [GitHub](https://github.com/LampLighterLab/BigRedGym). Notice that we
    must specify the branch as `main`, along with the repository itself, so we can clone the right one. In the future,
    you should substitute the URL for your fork and whatever branch you are developing policies on.
    """)
    return


@app.cell
def _():
    # --- edit, run this cell, then run the next one --------------------------
    BRANCH = "main"
    REPO_URL = "https://github.com/LampLighterLab/BigRedGym.git"
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
        uv python install 3.11          # pyproject pins requires-python == 3.11.*
        # --extra gpu pulls mujoco-warp (CUDA); --no-dev skips pytest/ruff/marimo.
        uv sync --frozen --extra gpu --no-dev
    """)
    # check=True: this ensures that the environment was built properly, so we know we're good to proceed.
    subprocess.run(["bash", "-c", _script], check=True, env=ENV)
    print("\nsetup done")
    return


@app.cell
def _(ENV, REPO, subprocess):
    # Sanity check: imports resolve and the GPU is visible. If `cuda` is False,
    # make sure that you are using the right GPU runtime. If you have questions
    # on this, please post on ED.
    _check = (
        "import torch, mujoco, mujoco_warp; "
        "print('torch', torch.__version__, '| mujoco', mujoco.__version__, "
        "'| cuda', torch.cuda.is_available(), "
        "'|', torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'NO GPU')"
    )
    subprocess.run(
        [
            "uv", "run", "--frozen", "--extra", "gpu", "--no-dev",
            "python", "-c", _check,
        ],
        cwd=REPO,
        env=ENV,
        check=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 Train

    Here, as this is running on GPU, we use `MuJoCo-Warp`, which makes use of the parallel
    nature of the GPU that the server comes with, in this case an RTX Blackwell 6000, with over 96 GB of VRAM!

    Below, we specify the task that we actually want to train on. Notice we have to specify
    the task *and* the device to train on.
    """)
    return


@app.cell
def _():
    # --- edit, run this cell, then **run the next one** --------------------------
    TASK = "go2trot"  # go2trot | mini_cheetah | humanoid | pendulum ...
    DEVICE = "cuda:0" # specifies which GPU to use - do not edit this line unless you know what this means!
    NUM_ENVS = 4096 # we specify multiples of 2 to compensate for the hardware.
    MAX_ITERATIONS = 550
    EXPERIMENT_NAME = ""  # blank -> the task's default
    RESUME = False  # resume the newest run for this experiment (if you stopped a training run)
    USE_WANDB = False  # don't worry about this for now, we'll go over it later in the course.
    return (
        DEVICE,
        EXPERIMENT_NAME,
        MAX_ITERATIONS,
        NUM_ENVS,
        RESUME,
        TASK,
        USE_WANDB,
    )


@app.cell
def _(
    DEVICE,
    ENV,
    EXPERIMENT_NAME,
    MAX_ITERATIONS,
    NUM_ENVS,
    REPO,
    RESUME,
    TASK,
    USE_WANDB,
    subprocess,
):
    # --- run this cell to train the policy! notice the outputs, what do you think they mean? --------------------------
    _cmd = ["uv", "run", "--frozen", "--extra", "gpu", "--no-dev"]
    _cmd += ["scripts/train.py"]
    _cmd += ["--task", TASK, "--device", DEVICE, "--headless"]
    _cmd += ["--num_envs", str(int(NUM_ENVS))]
    _cmd += ["--max_iterations", str(int(MAX_ITERATIONS))]
    if EXPERIMENT_NAME.strip():
        _cmd += ["--experiment_name", EXPERIMENT_NAME.strip()]
    if RESUME:
        _cmd += ["--resume"]
    if not USE_WANDB:
        _cmd += ["--disable_wandb"]

    print(" ".join(_cmd), "\n", flush=True)
    # Streamed line by line rather than captured, so the log appears while
    # training runs instead of all at once at the end.
    _proc = subprocess.Popen(
        _cmd,
        cwd=REPO,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for _line in _proc.stdout:
        print(_line, end="")
    if _proc.wait() != 0:
        raise RuntimeError(f"training exited with code {_proc.returncode}")
    print("\ntraining finished")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 Download checkpoints

    Runs live at `logs/<experiment_name>/<Mon##_HH-MM-SS>_<run>/model_<iter>.pt`.
    The zip carries each run's saved configs next to its checkpoints, so
    `--resume` and playback work from the extracted copy.

    Nothing here depends on section 2, so you can zip a run while training
    continues -- or in a runtime where the checkpoints are already on disk.
    Re-run the listing cell to pick up checkpoints written since you last ran it.
    """)
    return


@app.cell
def _(Path, re):
    def iteration(p: Path) -> int:
        return int(re.search(r"model_(\d+)\.pt", p.name).group(1))

    def find_runs(logs: Path) -> list[Path]:
        """Run dirs holding checkpoints, newest-touched first."""
        return sorted(
            {p.parent for p in logs.glob("*/*/model_*.pt")},
            key=lambda d: max(p.stat().st_mtime for p in d.glob("model_*.pt")),
            reverse=True,
        )

    return find_runs, iteration


@app.cell
def _(REPO, find_runs, iteration):
    runs = find_runs(REPO / "logs")
    if not runs:
        print(f"no checkpoints under {REPO / 'logs'} yet -- train first")
    for _run in runs:
        _ckpts = sorted(_run.glob("model_*.pt"), key=iteration)
        print(
            f"{_run.parent.name}/{_run.name}: {len(_ckpts)} ckpt, "
            f"iters {iteration(_ckpts[0])}..{iteration(_ckpts[-1])}"
        )
    return (runs,)


@app.cell
def _():
    # --- edit, run this cell, then run the next one --------------------------
    # None -> the newest run only. Otherwise a list of the `experiment/run`
    # names printed above, e.g. ["go2trot/Sep10_12-00-00_go2trot"].
    RUNS_TO_ZIP = None
    return (RUNS_TO_ZIP,)


@app.cell
def _(Path, RUNS_TO_ZIP, WORK, iteration, mo, runs, zipfile):
    if not runs:
        raise RuntimeError("no checkpoints to zip -- run section 2 first")
    if RUNS_TO_ZIP is None:
        _picked = runs[:1]
    else:
        _wanted = set(RUNS_TO_ZIP)
        _picked = [r for r in runs if f"{r.parent.name}/{r.name}" in _wanted]
        _missing = _wanted - {f"{r.parent.name}/{r.name}" for r in _picked}
        if _missing:
            raise RuntimeError(f"no such run(s): {sorted(_missing)}")

    _name = f"{_picked[0].name}.zip" if len(_picked) == 1 else "qgym_checkpoints.zip"
    _zip = WORK / "exports" / _name
    _zip.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoints are already-compressed tensors; ZIP_STORED keeps this fast.
    with zipfile.ZipFile(_zip, "w", compression=zipfile.ZIP_STORED) as _zf:
        for _run in _picked:
            for _f in sorted(_run.rglob("*")):
                if _f.is_file():
                    _arc = Path(_run.parent.name) / _run.name / _f.relative_to(_run)
                    _zf.write(_f, _arc)
            _ckpts = sorted(_run.glob("model_*.pt"), key=iteration)
            print(
                f"added {_run.parent.name}/{_run.name}: {len(_ckpts)} ckpt, "
                f"iters {iteration(_ckpts[0])}..{iteration(_ckpts[-1])}"
            )
    print(f"\n{_zip.name} -- {_zip.stat().st_size / 1e6:.1f} MB")

    # Lazy read: the archive is only slurped into memory when you click.
    mo.download(
        data=lambda p=_zip: p.read_bytes(), filename=_zip.name, label="Download"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Now you've trained a policy, and downloaded its log! Now try and running `play.py` locally on your machine - how does it look?
    What can you improve?
    """)
    return


if __name__ == "__main__":
    app.run()
