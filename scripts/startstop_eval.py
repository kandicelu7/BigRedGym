"""Start/stop test: stand, start, turn, stop and restart on command.

Every robot starts at rest and follows one command sequence, with forward
speed changes ramped at 2 m/s^2 (as the deploy settings train with):

    0-3 s    stop (command 0): stand
    3-9 s    start: ramp to the test speed v and hold
    9-12 s   turn: v with a yaw-rate command of 0.5 rad/s (11-12 s: 0 again)
    12-17 s  stop: ramp to 0 and stand
    17-21 s  restart: ramp to v and hold

at test speeds 1 and 2.5 m/s and moving oscillator frequencies 3 and 4.5 Hz
(policies with osc.stand_omega switch to their standing frequency
themselves while stopped). Measured per robot:

- fell: the base or head touched the ground (any phase).
- drift [m/s]: mean horizontal base speed while standing (1-3 s and
  15-17 s), and joint_speed [rad/s]: RMS joint speed there.
- steps [1/s]: touchdowns per second per active foot while standing
  (stepping in place).
- start_time / restart_time [s]: from the start command until the forward
  speed (0.2 s mean) first reaches 90% of v.
- stop_time [s]: from the stop command until the forward speed stays below
  0.1 m/s for 0.5 s; stop_distance [m]: distance covered meanwhile.
- yaw_ratio: mean yaw rate over 10-11 s / commanded 0.5 rad/s.

One policy:
    uv run --frozen scripts/startstop_eval.py --task go2osc_wide_even2.5_deploy --load_run <run>
Every run of some wide-study settings, then the summary:
    uv run --frozen scripts/startstop_eval.py --study --settings even2.5_deploy even2.5_robust baseline
    uv run --frozen scripts/startstop_eval.py --summary
Results: logs/study_results/startstop/<task>__<run>.npz; summary table in
results/study/startstop/.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SPEEDS = [1.0, 2.5]  # [m/s]
OMEGAS = [3.0, 4.5]  # [Hz]
ACCEL = 2.0  # [m/s^2]
YAW_RATE = 0.5  # [rad/s]
# Phase start times [s] and the end of the test.
T_START, T_TURN, T_TURN_END, T_STOP, T_RESTART, T_END = 3.0, 9.0, 11.0, 12.0, 17.0, 21.0
STAND_WINDOWS = [(1.0, 3.0), (15.0, 17.0)]
YAW_WINDOW = (10.0, 11.0)
TOUCH = 0.05  # foot load [x body weight] counted as contact
METRICS = [
    "fell",
    "drift",
    "joint_speed",
    "steps",
    "start_time",
    "stop_time",
    "stop_distance",
    "restart_time",
    "yaw_ratio",
]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--robots", type=int, default=64, help="robots per cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_out", type=str, default=None)
    parser.add_argument("--study", action="store_true", help="evaluate study runs")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="wide-study settings for --study (default: all)",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def command_profile(t, v):
    """Forward-speed and yaw-rate command at time t [s] for test speed v."""
    if t < T_START:
        vx = 0.0
    elif t < T_STOP:
        vx = min(v, ACCEL * (t - T_START))
    elif t < T_RESTART:
        vx = max(0.0, v - ACCEL * (t - T_STOP))
    else:
        vx = min(v, ACCEL * (t - T_RESTART))
    yaw = YAW_RATE if T_TURN <= t < T_TURN_END else 0.0
    return vx, yaw


def first_reach(vx, t0, v, dt, frac=0.9, smooth=0.2):
    """Per robot: seconds from t0 until the smoothed speed reaches frac*v."""
    k = max(1, int(smooth / dt))
    s = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, "same"), 1, vx)
    i0 = int(t0 / dt)
    out = np.full(vx.shape[0], np.nan)
    for r in range(vx.shape[0]):
        hit = np.flatnonzero(s[r, i0:] >= frac * v)
        if hit.size:
            out[r] = hit[0] * dt
    return out


def settle_below(vx, t0, t1, dt, limit=0.1, hold=0.5):
    """Per robot: seconds from t0 until |vx| stays below limit for `hold`
    seconds (within t0..t1), and the distance covered until then."""
    i0, i1, h = int(t0 / dt), int(t1 / dt), int(hold / dt)
    times = np.full(vx.shape[0], np.nan)
    dist = np.full(vx.shape[0], np.nan)
    for r in range(vx.shape[0]):
        below = np.abs(vx[r, i0:i1]) < limit
        run = np.convolve(below.astype(int), np.ones(h, int), "valid")
        hit = np.flatnonzero(run == h)
        if hit.size:
            times[r] = hit[0] * dt
            dist[r] = np.sum(vx[r, i0 : i0 + hit[0]]) * dt
    return times, dist


def rollout(env, runner, speed, omega):
    """The command sequence at one test speed and moving frequency."""
    env.reset()
    env.episode_omega[:] = omega
    env.osc_omega[:] = omega
    dt = env.dt
    n = int(round(T_END / dt))
    active = env.active_legs.cpu().numpy()
    vx = np.zeros((env.num_envs, n))
    vxy = np.zeros((env.num_envs, n))
    yaw_rate = np.zeros((env.num_envs, n))
    joint = np.zeros((env.num_envs, n))
    contact = np.zeros((env.num_envs, n, int(active.sum())), bool)
    fell = np.zeros(env.num_envs, bool)
    for i in range(n):
        cx, cyaw = command_profile(i * dt, speed)
        env.commands[:] = 0.0
        env.commands[:, 0] = cx
        env.commands[:, 2] = cyaw
        if hasattr(env, "command_target"):
            env.command_target[:] = cx  # the ramp is this script's
        env.episode_omega[:] = omega
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        fell |= env.terminated.cpu().numpy()
        v = env.base_lin_vel[:, :2].cpu().numpy()
        vx[:, i] = v[:, 0]
        vxy[:, i] = np.hypot(v[:, 0], v[:, 1])
        yaw_rate[:, i] = env.base_ang_vel[:, 2].cpu().numpy()
        joint[:, i] = (
            env.dof_vel[:, env.actuated_dof_indices].square().mean(1).cpu().numpy()
        )
        contact[:, i] = env.grf.cpu().numpy()[:, active] > TOUCH
    stand = np.zeros(n, bool)
    for a, b in STAND_WINDOWS:
        stand[int(a / dt) : int(b / dt)] = True
    touchdowns = (contact[:, 1:] & ~contact[:, :-1])[:, stand[1:]].sum((1, 2))
    stop_time, stop_dist = settle_below(vx, T_STOP, T_RESTART, dt)
    y0, y1 = int(YAW_WINDOW[0] / dt), int(YAW_WINDOW[1] / dt)
    out = {
        "fell": fell,
        "drift": vxy[:, stand].mean(1),
        "joint_speed": np.sqrt(joint[:, stand].mean(1)),
        "steps": touchdowns / (stand.sum() * dt * active.sum()),
        "start_time": first_reach(vx, T_START, speed, dt),
        "stop_time": stop_time,
        "stop_distance": stop_dist,
        "restart_time": first_reach(vx, T_RESTART, speed, dt),
        "yaw_ratio": yaw_rate[:, y0:y1].mean(1) / YAW_RATE,
    }
    for k in METRICS[1:]:
        out[k] = np.where(fell, np.nan, out[k])
    return out


def evaluate(args):
    import torch

    import play

    import gym.envs.go2.go2osc_config  # noqa: F401  (registers the study tasks)

    play_argv = ["--task", args.task, "--headless", "--no-keyboard", "--no-record"]
    play_argv += ["--num_envs", str(args.robots), "--seed", str(args.seed)]
    play_argv += ["--checkpoint", str(args.checkpoint), "--device", args.device]
    if args.load_run is not None:
        play_argv += ["--load_run", args.load_run]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
    out = {k: [] for k in METRICS}
    for omega in OMEGAS:
        rows = {k: [] for k in METRICS}
        for speed in SPEEDS:
            with torch.no_grad():
                r = rollout(env, runner, speed, omega)
            for k in METRICS:
                rows[k].append(r[k])
            print(
                f"{args.task} v{speed:g} w{omega:g}: falls {r['fell'].mean():.0%}, "
                f"drift {np.nanmean(r['drift']):.2f} m/s, steps {np.nanmean(r['steps']):.2f}/s, "
                f"start {np.nanmedian(r['start_time']):.2f} s, stop {np.nanmedian(r['stop_time']):.2f} s, "
                f"yaw {np.nanmean(r['yaw_ratio']):.2f}",
                flush=True,
            )
        for k in METRICS:
            out[k].append(np.stack(rows[k]))
    path = Path(args.data_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        speeds=np.array(SPEEDS),
        omegas=np.array(OMEGAS),
        run=str(load_path),
        **{k: np.stack(v) for k, v in out.items()},
    )  # [omega, speed, robot]
    print(f"Saved {path}")


def paths():
    from study_summary import DATA, FIGURES

    return DATA / "startstop", FIGURES / "startstop"


def run_study(args):
    from study_eval import study_runs

    data_dir, _ = paths()
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    from gym.envs.go2.go2osc_config import HARDWARE_TASKS

    tasks = (
        None
        if args.settings is None
        else {s if s in HARDWARE_TASKS else f"go2osc_wide_{s}" for s in args.settings}
    )
    todo = []
    for task, run in study_runs(True):
        if tasks is not None and task not in tasks:
            continue
        out = data_dir / f"{task}__{run}.npz"
        if out.exists() and not args.force:
            print(f"done    {out.name}")
        else:
            todo.append((task, run, out))
    print(f"{len(todo)} to evaluate", flush=True)
    from eval_progress import JobPool, interval_from_env, unbuffered

    env = unbuffered(dict(os.environ, OMP_NUM_THREADS=str(args.threads)))
    pool = JobPool(args.jobs, len(SPEEDS) * len(OMEGAS), interval_from_env())
    for task, run, out in todo:
        cmd = [sys.executable, __file__, "--task", task, "--load_run", run]
        cmd += ["--checkpoint", str(args.checkpoint), "--robots", str(args.robots)]
        cmd += ["--data_out", str(out), "--device", args.device]
        log_path = logs / f"{out.stem}.log"
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            env=env,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        pool.add(proc, out.name, log_path)
    pool.close()
    if pool.failed:
        raise SystemExit(f"failed (see {logs}): {pool.failed}")


def summarize(args):
    data_dir, fig_dir = paths()
    groups = {}
    for path in sorted(data_dir.glob("go2osc_wide_*.npz")):
        setting = path.name.split("__")[0].removeprefix("go2osc_wide_")
        d = np.load(path, allow_pickle=False)
        stats = {
            k: (
                float(np.mean(d[k]))
                if k == "fell"
                else float(np.nanmedian(d[k]))
                if np.isfinite(d[k]).any()
                else np.nan
            )
            for k in METRICS
        }
        # Share of robots (that did not fall) that came to a stop at all;
        # stop_time and stop_distance are medians over those.
        stats["stopped"] = float(np.isfinite(d["stop_time"])[~d["fell"]].mean())
        groups.setdefault(setting, []).append(stats)
    if not groups:
        raise SystemExit(f"no results in {data_dir}; run --study first")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    keys = METRICS + ["stopped"]
    print(f"{'setting':22s} {'runs':>4s} " + " ".join(f"{k:>13s}" for k in keys))
    for setting, runs in groups.items():
        row = {"setting": setting, "runs": len(runs)}
        for k in keys:
            v = np.array([r[k] for r in runs])
            row[f"{k}_mean"], row[f"{k}_sd"] = np.nanmean(v), np.nanstd(v)
        rows.append(row)
        print(
            f"{setting:22s} {len(runs):4d} "
            + " ".join(f"{row[f'{k}_mean']:6.2f}±{row[f'{k}_sd']:5.2f}" for k in keys)
        )
    path = fig_dir / "startstop_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}  (fell, stopped: share of robots; others: per-run medians)")


def main():
    args = get_args()
    if args.summary:
        summarize(args)
    elif args.study:
        run_study(args)
    else:
        if args.data_out is None:
            args.data_out = str(paths()[0] / f"{args.task}__{args.load_run}.npz")
        evaluate(args)


if __name__ == "__main__":
    main()
