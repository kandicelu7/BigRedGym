# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Study training queue on MoLab, with every finished run backed up to Hugging Face.

Like molab_train.py, but it trains a list of (task, seed) runs in turn and,
after each one, uploads it to a private Hugging Face repo with
scripts/hf_runs.py. If molab reclaims the session you lose at most the run in
progress: rerun the notebook and the queue skips every run already stored.

On your PC, pull the runs into logs/ with

    uv run --no-project --with huggingface_hub python scripts/hf_runs.py fetch

Run the cells in order, top to bottom.
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
    from pathlib import Path

    import marimo as mo

    return Path, mo, os, re, subprocess, textwrap


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Study queue with Hugging Face backup

    1. **Setup**: clone or update the repo and install the dependencies.
    2. **Token**: paste a Hugging Face *write* token and check it.
    3. **Queue**: train each run; each one is uploaded as soon as it finishes.

    One-time: make a free account at huggingface.co, then create a token
    under Settings → Access Tokens with **Write** permission.
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
    _shadowing = ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME",
                  "PYTHONEXECUTABLE")
    ENV = {
        k: v
        for k, v in os.environ.items()
        if k not in _shadowing and not (k.startswith("UV_") and k != "UV_CACHE_DIR")
    }
    ENV["UV_PYTHON"] = "3.11"
    print(f"repo: {REPO}")
    return ENV, REPO


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
        uv python install 3.11
        uv sync --frozen --extra gpu --no-dev
    """)
    subprocess.run(["bash", "-c", _script], check=True, env=ENV)
    print("\nsetup done")
    return


@app.cell
def _(mo):
    # Paste the token here, then run the next cell. It is held in memory only,
    # never saved into the notebook file.
    token = mo.ui.text(kind="password", label="Hugging Face write token", full_width=True)
    token
    return (token,)


@app.cell
def _(ENV, REPO, os, subprocess, token):
    HF_ENV = {**ENV, "HF_TOKEN": token.value.strip() or os.environ.get("HF_TOKEN", "")}
    if not HF_ENV["HF_TOKEN"]:
        raise RuntimeError("paste your Hugging Face write token above, then rerun this cell")
    HF = ["uv", "run", "--no-project", "--with", "huggingface_hub", "python",
          "scripts/hf_runs.py"]
    # Creates the private repo if needed and shows what it already holds.
    subprocess.run(HF + ["check"], cwd=REPO, env=HF_ENV, check=True)
    _stored = subprocess.run(HF + ["list"], cwd=REPO, env=HF_ENV, check=True,
                             capture_output=True, text=True).stdout
    STORED = _stored.split()
    print(_stored or "(no runs stored yet)")
    return HF, HF_ENV, STORED


@app.cell
def _():
    DEVICE = "cuda:0"
    NUM_ENVS = 4096
    MAX_ITERATIONS = 1500
    # Follow-up batches of the wide study (go2osc_config.FOLLOWUP_BATCHES):
    #   1: robust, even2.5_robust, even2.5_smooth0.5, even2.5_smooth1
    #   2: push, friction, even2.5_noflight1, even2.5_cot0.2
    #   3: even2.5_deploy, even2.5_track2
    #   4: even2.5_deploy_robust
    #   5: even2.5_deploy_ft4 (fine-tunes deploy seed 4), even2.5_deploy_track
    #   6: deploy_robust (sim-to-real package on the baseline rewards)
    #   7: deploy_robust_clip_ft2, even2.5_deploy_track_clip_ft2 (fine-tunes
    #      under 95% actuator limits; MAX_ITERATIONS = 1000)
    # PART splits a batch over two notebooks (0 and 1); None runs all of it.
    BATCH = 1
    PART = None
    _BATCHES = {  # as in go2osc_config.FOLLOWUP_BATCHES
        1: ["robust", "even2.5_robust", "even2.5_smooth0.5", "even2.5_smooth1"],
        2: ["push", "friction", "even2.5_noflight1", "even2.5_cot0.2"],
        3: ["even2.5_deploy", "even2.5_track2"],
        4: ["even2.5_deploy_robust"],
        5: ["even2.5_deploy_ft4", "even2.5_deploy_track"],
        6: ["deploy_robust"],
        7: ["deploy_robust_clip_ft2", "even2.5_deploy_track_clip_ft2"],
    }
    SETTINGS = _BATCHES[BATCH]
    # Rounds of seeds, so every setting keeps the same seed count if the
    # queue stops early.
    QUEUE = [(f"go2osc_wide_{s}", _k) for _k in range(5) for s in SETTINGS]
    if PART is not None:
        QUEUE = QUEUE[PART::2]
    print(f"batch {BATCH}: {SETTINGS}")
    print(f"{len(QUEUE)} runs, about {len(QUEUE) * 14 / 60:.1f} h")
    return DEVICE, MAX_ITERATIONS, NUM_ENVS, QUEUE


@app.cell
def _(DEVICE, ENV, HF, HF_ENV, MAX_ITERATIONS, NUM_ENVS, QUEUE, REPO, STORED,
      re, subprocess):
    def _done(task, seed):
        """Finished here, or already stored on Hugging Face from an earlier session."""
        local = any((REPO / "logs" / task).glob(f"*_seed{seed}/model_{MAX_ITERATIONS}.pt"))
        remote = any(re.fullmatch(rf"{re.escape(task)}/[^/]+_seed{seed}", s) for s in STORED)
        return local or remote

    def _sync():
        _r = subprocess.run(HF + ["sync", "--iterations", str(MAX_ITERATIONS)],
                            cwd=REPO, env=HF_ENV, capture_output=True, text=True)
        print(_r.stdout + _r.stderr[-1500:], end="", flush=True)
        if _r.returncode:
            print("!!! upload failed; it is retried after the next run", flush=True)

    for _i, (_task, _seed) in enumerate(QUEUE, 1):
        if _done(_task, _seed):
            print(f"skip {_task} seed{_seed}: already trained", flush=True)
            continue
        print(f"\n=== [{_i}/{len(QUEUE)}] {_task} seed{_seed}", flush=True)
        _cmd = ["uv", "run", "--frozen", "--extra", "gpu", "--no-dev", "scripts/train.py",
                "--task", _task, "--seed", str(_seed), "--run_name", f"seed{_seed}",
                "--device", DEVICE, "--headless", "--disable_wandb",
                "--num_envs", str(int(NUM_ENVS)),
                "--max_iterations", str(int(MAX_ITERATIONS))]
        _proc = subprocess.Popen(_cmd, cwd=REPO, env={**ENV, "PYTHONUNBUFFERED": "1"},
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for _line in _proc.stdout:
            _m = re.search(r"Iteration (\d+)/(\d+)", _line)
            if _m and int(_m.group(1)) % 250 == 0:
                print(f"  iteration {_m.group(1)}/{_m.group(2)}", flush=True)
            elif "Error" in _line or "Traceback" in _line:
                print(_line, end="", flush=True)
        if _proc.wait() != 0:
            print(f"!!! {_task} seed{_seed} failed with code {_proc.returncode}", flush=True)
        _sync()

    print("\nqueue finished")
    return


if __name__ == "__main__":
    app.run()
