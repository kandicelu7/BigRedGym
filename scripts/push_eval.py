"""Push test: shove each robot once and measure whether and how fast it recovers.

Training pushes were tiny (at most 0.05 m/s), so the grid evaluation runs
without them. Here every robot walks at a commanded speed and oscillator
frequency, settles, and gets one horizontal push: its body velocity changes
instantly by a set amount (0.5, 1.0, 1.5 or 2.0 m/s) forward, backward, to the
left or to the right of its heading. With 64 robots per cell that is 4 robots
per (direction, size). Measured over the following seconds:

- fell: the base or head touched the ground (the episode terminated).
- recovery time: until the robot's speed, averaged over half a second, is
  back within 0.25 m/s of its own pre-push average, forward and sideways,
  for at least half a second. NaN if it fell or never got back.
- heading change: extra turning in the second after the push, beyond the
  robot's own turning in the second before it [deg].

Robots that fell before the push are left out.

One policy:
    uv run --frozen scripts/push_eval.py --task go2osc_study_baseline --load_run <run>
Every study run (several at once, skipping done ones), then the summary:
    uv run --frozen scripts/push_eval.py --study [--wide] [--jobs 4]
    uv run --frozen scripts/push_eval.py --summary [--wide]
Results: logs/study_results/push/<task>__<run>.npz; summary table and figure
in results/study/push/ (or push_wide/).
"""

import argparse
import csv
import os
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

DIRECTIONS = ["forward", "backward", "left", "right"]
# Unit push direction in the robot's heading frame (x forward, y left).
DIRECTION_VECTORS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
MAGNITUDES = [0.5, 1.0, 1.5, 2.0]  # [m/s]
# Cells: frequencies where the policies hop (1.5 Hz is irregular for all).
STUDY_SPEEDS, STUDY_OMEGAS = [0.5, 1.5, 2.5], [2.25, 3.0, 3.75]
WIDE_SPEEDS, WIDE_OMEGAS = [1.0, 2.5, 4.0], [3.0, 4.5]
RECOVERY_TOL = 0.25  # [m/s]
RECOVERY_WINDOW = 0.5  # [s]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Push test")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--robots", type=int, default=64, help="robots per cell")
    parser.add_argument("--settle", type=float, default=5.0, help="before the push [s]")
    parser.add_argument("--after", type=float, default=4.0, help="after the push [s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--data_out", type=str, default=None)
    parser.add_argument("--study", action="store_true", help="every study run")
    parser.add_argument("--summary", action="store_true", help="summarize results")
    parser.add_argument("--wide", action="store_true", help="the wide-range study")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="redo done runs")
    return parser.parse_args(argv)


def assignment(n_robots):
    """(direction index, magnitude) of each robot."""
    direction = np.arange(n_robots) % len(DIRECTIONS)
    magnitude = np.array(MAGNITUDES)[(np.arange(n_robots) // len(DIRECTIONS)) % len(MAGNITUDES)]
    return direction, magnitude


def centered_mean(x, w):
    """Mean of x[:, t - w//2 : t + w//2 + 1] for every t (edges shrink)."""
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0))), axis=1)
    n = x.shape[1]
    lo = np.clip(np.arange(n) - w // 2, 0, n)
    hi = np.clip(np.arange(n) + w // 2 + 1, 0, n)
    return (c[:, hi] - c[:, lo]) / (hi - lo)


def recovery_times(vx, vy, pre_vx, pre_vy, dt):
    """First time after the push from which the half-second mean velocity
    stays within RECOVERY_TOL of the pre-push mean for RECOVERY_WINDOW [s]."""
    w = int(round(RECOVERY_WINDOW / dt))
    ok = (np.abs(centered_mean(vx, w) - pre_vx[:, None]) < RECOVERY_TOL) & (
        np.abs(centered_mean(vy, w) - pre_vy[:, None]) < RECOVERY_TOL
    )
    # ok for the whole window starting at t
    run = np.zeros_like(ok, dtype=int)
    for t in range(ok.shape[1] - 1, -1, -1):
        run[:, t] = np.where(ok[:, t], 1 + (run[:, t + 1] if t + 1 < ok.shape[1] else 0), 0)
    out = np.full(vx.shape[0], np.nan)
    for r in range(vx.shape[0]):
        hit = np.flatnonzero(run[r] >= w)
        if hit.size:
            out[r] = hit[0] * dt
    return out


def push_rollout(env, runner, speed, omega, args):
    """One push per robot after settling; returns per-robot results."""
    import torch

    from grid_eval import heading

    env.reset()
    env.episode_omega[:] = omega
    env.osc_omega[:] = omega
    direction, magnitude = assignment(env.num_envs)
    n_settle = int(args.settle / env.dt)
    n_after = int(args.after / env.dt)
    n_pre = int(2.0 / env.dt)
    n_turn = int(1.0 / env.dt)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    fell_before = np.zeros(env.num_envs, bool)
    fell_after = np.zeros(env.num_envs, bool)
    vx_pre, vy_pre, vx, vy = [], [], [], []
    for i in range(n_settle + n_after):
        env.commands[:] = 0.0
        env.commands[:, 0] = speed
        env.episode_omega[:] = omega
        if i == n_settle - n_turn:
            yaw_before = heading(env)
        if i == n_settle:
            yaw0 = heading(env)
            c, s = np.cos(yaw0), np.sin(yaw0)
            local = DIRECTION_VECTORS[direction] * magnitude[:, None]
            world = np.stack(
                [c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]], 1
            )
            env.root_states[:, 7:9] += torch.as_tensor(
                world, dtype=env.root_states.dtype, device=env.root_states.device
            )
            env._backend.set_all_root_states()
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        if i == n_settle + n_turn - 1:
            turned = wrap(heading(env) - yaw0) - wrap(yaw0 - yaw_before)
        terminated = env.terminated.cpu().numpy()
        if i < n_settle:
            fell_before |= terminated
        else:
            fell_after |= terminated & ~fell_before
        v = env.base_lin_vel[:, :2].cpu().numpy().copy()
        if n_settle - n_pre <= i < n_settle:
            vx_pre.append(v[:, 0])
            vy_pre.append(v[:, 1])
        elif i >= n_settle:
            vx.append(v[:, 0])
            vy.append(v[:, 1])
    recovery = recovery_times(
        np.stack(vx, 1),
        np.stack(vy, 1),
        np.mean(vx_pre, 0),
        np.mean(vy_pre, 0),
        env.dt,
    )
    recovery[fell_after | fell_before] = np.nan
    return {
        "direction": direction,
        "magnitude": magnitude,
        "fell_before": fell_before,
        "fell": fell_after,
        "recovery_time": recovery,
        "heading_change": np.degrees(np.abs(turned)),
    }


def evaluate(args):
    import torch

    import play

    from gym.envs.go2.go2osc_config import study_task_name  # noqa: F401  (registers)

    wide = args.task.startswith("go2osc_wide_")
    speeds = WIDE_SPEEDS if wide else STUDY_SPEEDS
    omegas = WIDE_OMEGAS if wide else STUDY_OMEGAS
    play_argv = ["--task", args.task, "--headless", "--no-keyboard", "--no-record"]
    play_argv += ["--num_envs", str(args.robots), "--seed", str(args.seed)]
    play_argv += ["--checkpoint", str(args.checkpoint), "--device", args.device]
    if args.load_run is not None:
        play_argv += ["--load_run", args.load_run]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
    keys = ["direction", "magnitude", "fell_before", "fell", "recovery_time", "heading_change"]
    out = {k: [] for k in keys}
    for omega in omegas:
        rows = {k: [] for k in keys}
        for speed in speeds:
            with torch.no_grad():
                r = push_rollout(env, runner, speed, omega, args)
            for k in keys:
                rows[k].append(r[k])
            kept = ~r["fell_before"]
            print(
                f"{args.task} v{speed:g} w{omega:g}: falls "
                + " ".join(
                    f"{m:g}:{r['fell'][kept & (r['magnitude'] == m)].mean():.0%}"
                    for m in MAGNITUDES
                ),
                flush=True,
            )
        for k in keys:
            out[k].append(np.stack(rows[k]))
    data = {k: np.stack(v) for k, v in out.items()}  # [omega, speed, robot]
    path = Path(args.data_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        speeds=np.array(speeds),
        omegas=np.array(omegas),
        magnitudes=np.array(MAGNITUDES),
        directions=np.array(DIRECTIONS),
        run=str(load_path),
        **data,
    )
    print(f"Saved {path}")


def paths(wide):
    from study_summary import DATA, FIGURES

    return DATA / ("push_wide" if wide else "push"), FIGURES / (
        "push_wide" if wide else "push"
    )


def run_study(args):
    from study_eval import study_runs

    data_dir, _ = paths(args.wide)
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = []
    for task, run in study_runs(args.wide):
        out = data_dir / f"{task}__{run}.npz"
        if out.exists() and not args.force:
            print(f"done    {out.name}")
        else:
            todo.append((task, run, out))
    print(f"{len(todo)} to evaluate", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    running, failed = [], []

    def finish(entry):
        proc, name = entry
        if proc.wait() != 0:
            failed.append(name)
        print(f"{'FAILED' if proc.returncode else 'done  '}  {name}", flush=True)

    for task, run, out in todo:
        cmd = [sys.executable, __file__, "--task", task, "--load_run", run]
        cmd += ["--checkpoint", str(args.checkpoint), "--robots", str(args.robots)]
        cmd += ["--data_out", str(out), "--device", args.device]
        log = open(logs / f"{out.stem}.log", "w")
        proc = subprocess.Popen(
            cmd, cwd=Path(__file__).parent, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        running.append((proc, out.name))
        if len(running) >= args.jobs:
            finish(running.pop(0))
    for entry in running:
        finish(entry)
    if failed:
        raise SystemExit(f"failed (see {logs}): {failed}")


# ---------------------------------------------------------------- summary
def run_stats(d):
    """Per-run numbers: fall rate per magnitude, per direction (at 1.0 and
    1.5 m/s) and median recovery time / heading change per magnitude."""
    kept = ~d["fell_before"]
    mag, dirn = d["magnitude"], d["direction"]
    out = {}
    for m in MAGNITUDES:
        sel = kept & (mag == m)
        out[f"fall@{m:g}"] = d["fell"][sel].mean()
        rec = d["recovery_time"][sel & ~d["fell"]]
        out[f"recovered@{m:g}"] = np.isfinite(rec).mean() if rec.size else np.nan
        out[f"recovery_s@{m:g}"] = np.nanmedian(rec) if np.isfinite(rec).any() else np.nan
        out[f"turn_deg@{m:g}"] = np.median(d["heading_change"][sel & ~d["fell"]])
    for k, name in enumerate(DIRECTIONS):
        sel = kept & (dirn == k) & np.isin(mag, [1.0, 1.5])
        out[f"fall_{name}"] = d["fell"][sel].mean()
    return out


def summarize(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import study_summary
    from study_summary import INK, MUTED, permutation_p, style

    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    if args.wide:
        study_summary.use_wide()
    prefix, settings = study_summary.PREFIX, study_summary.STUDY_SETTINGS
    data_dir, fig_dir = paths(args.wide)
    runs = defaultdict(list)
    for path in sorted(data_dir.glob(f"{prefix}*.npz")):
        setting = path.name.split("__")[0].removeprefix(prefix)
        if setting in settings:
            runs[setting].append(run_stats(np.load(path, allow_pickle=True)))
    if not runs:
        raise SystemExit(f"no push results in {data_dir}; run --study first")
    keys = list(next(iter(runs.values()))[0])
    rows = []
    base = runs.get("baseline", [])
    for setting in settings:
        group = runs.get(setting)
        if not group:
            continue
        row = {"setting": setting, "seeds": len(group)}
        for k in keys:
            v = np.array([g[k] for g in group], float)
            row[f"{k}_mean"] = np.nanmean(v)
            row[f"{k}_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else np.nan
            if base and setting != "baseline":
                ref = np.array([g[k] for g in base], float)
                if not (np.isnan(v).any() or np.isnan(ref).any()):
                    row[f"{k}_p"], _ = permutation_p(
                        v, ref, lambda a, b: np.mean(a) - np.mean(b)
                    )
        rows.append(row)
    fig_dir.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(fig_dir / "push_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(r, k, pct=True):
        m, s, p = r[f"{k}_mean"], r[f"{k}_sd"], r.get(f"{k}_p", 1.0)
        txt = f"{m:.0%}±{s:.0%}" if pct else f"{m:.2f}±{s:.2f}"
        return f"{txt}{'*' if p < 0.05 else ' '}"

    print(f"\nfall rate by push size (mean±SD over seeds; * p < 0.05 against baseline)")
    print(f"{'setting':10s} seeds " + "".join(f"{f'{m:g} m/s':>12s}" for m in MAGNITUDES)
          + "".join(f"{d + ' (1-1.5)':>17s}" for d in DIRECTIONS))
    for r in rows:
        print(f"{r['setting']:10s} {r['seeds']:5d} "
              + "".join(f"{cell(r, f'fall@{m:g}'):>12s}" for m in MAGNITUDES)
              + "".join(f"{cell(r, f'fall_{d}'):>17s}" for d in DIRECTIONS))
    print(f"\nmedian recovery time [s] of robots that stayed up")
    print(f"{'setting':10s} " + "".join(f"{f'{m:g} m/s':>13s}" for m in MAGNITUDES))
    for r in rows:
        print(f"{r['setting']:10s} "
              + "".join(f"{cell(r, f'recovery_s@{m:g}', pct=False):>13s}" for m in MAGNITUDES))

    names = [r["setting"] for r in rows]
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, label) in zip(
        axes,
        [("fall", "fall rate"), ("recovery_s", "median recovery time [s]"),
         ("turn_deg", "median heading change [deg]")],
    ):
        style(ax)
        for i, r in enumerate(rows):
            y = [r[f"{key}@{m:g}_mean"] for m in MAGNITUDES]
            e = [r[f"{key}@{m:g}_sd"] for m in MAGNITUDES]
            ax.errorbar(
                np.array(MAGNITUDES) + (i - len(rows) / 2) * 0.012, y, yerr=e,
                marker="o", ms=4, lw=2 if r["setting"] == "baseline" else 1.2,
                color=INK if r["setting"] == "baseline" else colors(i % 10),
                label=r["setting"], capsize=2,
            )
        ax.set_xlabel("push size [m/s]", color=MUTED)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_xticks(MAGNITUDES)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Push test: one push per robot, mean ± SD over seeds", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "push_summary.png", dpi=120)

    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(names), 3.8))
    style(ax)
    width = 0.8 / len(DIRECTIONS)
    for k, d in enumerate(DIRECTIONS):
        ax.bar(np.arange(len(names)) + (k - 1.5) * width,
               [r[f"fall_{d}_mean"] for r in rows], width,
               yerr=[r[f"fall_{d}_sd"] for r in rows], label=d, capsize=1.5,
               color=["#2a78d6", "#8fb8ea", "#eb6834", "#f2b08f"][k])
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Fall rate by push direction (1.0 and 1.5 m/s pushes); "
                 "right = toward the locked front-right leg", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "push_directions.png", dpi=120)
    print(f"\nSaved {fig_dir / 'push_summary.csv'}, push_summary.png, push_directions.png")


if __name__ == "__main__":
    args = get_args()
    if args.summary:
        summarize(args)
    elif args.study:
        run_study(args)
    else:
        if args.task is None or args.data_out is None:
            raise SystemExit("--task and --data_out are needed for one policy")
        evaluate(args)
