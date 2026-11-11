"""Gait distribution over time, after Fig. 5 of Zhang et al. (2024).

Rolls out one policy with many robots from random initial oscillator phases,
all at one commanded speed and oscillator frequency. Every `--interval`
seconds each robot's relative phase difference (RPD: when each leg lands
within a stride of the reference leg) is averaged over its last two strides
and assigned to the nearest ideal gait by Euclidean distance in RPD space
[rad]. Farther than `--transition_rad` (2 rad in the paper) is "transition".
The share of robots in each gait is plotted against time as stacked areas.

Intact robots use the paper's gaits (trot, pace, bound, pronk) with the FR
foot as reference. Three-legged robots use the three-legged gaits of
grid_eval.py, with the first active leg as reference.

Usage:
    uv run --frozen scripts/gait_over_time.py --task go2osc_front_locked_omega --speed 1 --omega 2.5
    uv run --frozen scripts/gait_over_time.py --task go2osc --speed 1 --omega 2.5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import footfall
import play
from grid_eval import gait_reference
from gym.utils.helpers import GYM_ROOT_DIR

TRANSITION = "transition"
NOT_STEPPING = "not stepping"
FELL = "fell"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Gait distribution over time")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--load_run", type=str, default=None, help="default: latest")
    parser.add_argument("--checkpoint", type=int, default=-1, help="default: latest")
    parser.add_argument("--speed", type=float, default=1.0, help="command [m/s]")
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="oscillator frequency [Hz]; default: the task's own (per-episode "
        "draw or speed schedule). Forcing it on a policy trained with the "
        "schedule tests a frequency that policy never saw.",
    )
    parser.add_argument("--robots", type=int, default=500)
    parser.add_argument("--duration", type=float, default=40.0, help="[s]")
    parser.add_argument("--interval", type=float, default=5.0, help="[s]")
    parser.add_argument("--strides", type=int, default=2, help="strides averaged")
    parser.add_argument("--transition_rad", type=float, default=2.0)
    parser.add_argument("--contact_threshold", type=float, default=0.05)
    parser.add_argument("--min_air", type=float, default=0.03, help="[s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="figure path (.png)")
    return parser.parse_args(argv)


def gait_set(legs, leg_names):
    """Reference leg, compared legs, names and ideal RPDs [rad]."""
    ref, others, gaits = gait_reference(legs, leg_names)
    return ref, others, [(n, tuple(2 * np.pi * np.asarray(r))) for n, r in gaits]


def classify(rpd, gaits, transition_rad):
    if np.isnan(rpd).any():
        return NOT_STEPPING
    best, best_distance = None, np.inf
    for name, ideal in gaits:
        gap = np.abs(rpd - np.asarray(ideal)) % (2 * np.pi)
        distance = np.linalg.norm(np.minimum(gap, 2 * np.pi - gap))
        if distance < best_distance:
            best, best_distance = name, distance
    return best if best_distance <= transition_rad else TRANSITION


def last_strides_rpd(contact, ref, others, min_air, max_stride, strides):
    """RPD [rad] averaged (circularly) over the last `strides` ref strides."""
    starts, fractions = footfall.relative_touchdowns(
        contact, ref, others, min_air, max_stride
    )
    if len(starts) < strides:
        return np.full(len(others), np.nan)
    rpd = []
    for leg in others:
        # A landing jittered across the reference touchdown leaves one stride
        # without it (and the neighbour with two), so skip empty strides; a
        # leg that landed in none of them is not stepping.
        f = fractions[leg][-strides:]
        f = f[~np.isnan(f)]
        if f.size == 0:
            return np.full(len(others), np.nan)
        angle = np.angle(np.exp(2j * np.pi * f).mean())
        rpd.append(angle % (2 * np.pi))
    return np.array(rpd)


def rollout(env, runner, args, omega_mode):
    """Contact per step for every robot, and the step each robot first fell."""
    weight = 9.81 * env._backend.link_mass.sum(dim=1, keepdim=True)
    env.reset()
    if omega_mode == "episode":
        env.episode_omega[:] = args.omega
        env.osc_omega[:] = args.omega
    steps = int(round(args.duration / env.dt))
    contact = np.zeros((env.num_envs, steps, len(env.feet_indices)), dtype=bool)
    fell_at = np.full(env.num_envs, steps)
    for i in range(steps):
        env.commands[:] = 0.0
        env.commands[:, 0] = args.speed
        if omega_mode == "episode":
            env.episode_omega[:] = args.omega
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        fell = env.terminated.numpy()
        fell_at[fell & (fell_at == steps)] = i
        load = env.contact_forces[:, env.feet_indices, 2] / weight
        contact[:, i] = (load > args.contact_threshold).numpy()
    return contact, fell_at


def main():
    args = get_args()
    play_argv = ["--task", args.task, "--headless", "--no-keyboard", "--no-record"]
    play_argv += ["--num_envs", str(args.robots), "--seed", str(args.seed)]
    play_argv += ["--checkpoint", str(args.checkpoint)]
    if args.load_run is not None:
        play_argv += ["--load_run", args.load_run]

    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
        osc = env.cfg.osc
        if args.omega is None:
            omega_mode = "task"
            if osc.omega_range is not None:
                raise SystemExit(
                    f"{args.task} draws omega per episode; pass --omega"
                )
            omega_label = (
                f"{min(osc.omega_min + osc.omega_slope * abs(args.speed), osc.omega_max):g}"
                " (schedule)"
            )
        elif osc.omega_range is not None:
            omega_mode = "episode"
            omega_label = f"{args.omega:g}"
        else:
            omega_mode = "fixed"
            osc.omega_min, osc.omega_slope = args.omega, 0.0
            osc.omega_max = max(args.omega, osc.omega_max)
            omega_label = f"{args.omega:g} (forced; trained on schedule)"
        contact, fell_at = rollout(env, runner, args, omega_mode)

    feet = env.robot_layout.body_groups["feet"]
    legs = [k for k, active in enumerate(env.active_legs.tolist()) if active]
    leg_names = [feet[k].removesuffix("_foot") for k in legs]
    ref, others, gaits = gait_set(legs, leg_names)
    names = [name for name, _ in gaits] + [TRANSITION, NOT_STEPPING, FELL]

    min_air = int(round(args.min_air / env.dt))
    max_stride = int(round(2.0 / env.dt))
    times = np.arange(args.interval, args.duration + 1e-9, args.interval)
    shares = np.zeros((len(times), len(names)))
    for t, time in enumerate(times):
        end = min(int(round(time / env.dt)), contact.shape[1])
        for robot in range(env.num_envs):
            if fell_at[robot] < end:
                label = FELL
            else:
                rpd = last_strides_rpd(
                    contact[robot, :end], ref, others, min_air, max_stride, args.strides
                )
                label = classify(rpd, gaits, args.transition_rad)
            shares[t, names.index(label)] += 1
    shares /= env.num_envs

    run = f"{Path(load_path).parent.name}/{Path(load_path).stem}"
    out = Path(args.out) if args.out else (
        Path(GYM_ROOT_DIR) / "logs" / "gait_over_time"
        / f"{args.task}_v{args.speed:g}_w{args.omega if args.omega is not None else 'sched'}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out.with_suffix(".npz"),
        times=times, shares=shares, names=np.array(names), run=run,
        speed=args.speed, omega=np.nan if args.omega is None else args.omega,
    )

    print(f"{run}: {env.num_envs} robots, {args.speed:g} m/s, omega {omega_label} Hz")
    print("time  " + "  ".join(f"{n:>14s}" for n in names))
    for time, row in zip(times, shares):
        print(f"{time:4.0f}s " + "  ".join(f"{v:14.0%}" for v in row))

    # Gait colors fixed by position; transition / not stepping / fell in greys.
    colors = list(plt.cm.tab10(np.arange(len(gaits))))
    colors += [(0.75, 0.75, 0.75, 1), (0.45, 0.45, 0.45, 1), (0.15, 0.15, 0.15, 1)]
    keep = shares.max(axis=0) > 0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(
        times,
        shares[:, keep].T,
        labels=[n for n, k in zip(names, keep) if k],
        colors=[c for c, k in zip(colors, keep) if k],
    )
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% Experiments")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    ax.set_title(
        f"Gait distribution over time: {args.task} ({run})\n"
        f"omega = {omega_label} Hz, command = {args.speed:g} m/s, "
        f"{env.num_envs} robots from random phases",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
