"""Footfall (gait) diagram of a trained policy at several forward speeds.

Runs one headless env per commanded speed, records which feet are loaded each
control step, and plots time on the x axis against the unlocked legs on the
y axis: black where the foot is on the ground, empty otherwise.

Usage:
    uv run --frozen scripts/footfall.py --task go2osc_front_locked
    uv run --frozen scripts/footfall.py --task go2osc --speeds 0.5 1 2 3 \
        --load_run Oct30_21-51-01_ --checkpoint 500 --out trot.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import play


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot footfall sequences")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[0.75, 1.25, 2.5],
        help="commanded forward speeds [m/s], one panel each",
    )
    parser.add_argument(
        "--duration", type=float, default=15.0, help="simulated time [s]"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="plot only the last WINDOW seconds, after the gait has settled",
    )
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=0.05,
        help="vertical foot force, as a fraction of body weight, counted as contact",
    )
    parser.add_argument("--out", type=str, default=None, help="figure path")
    return parser.parse_args(argv)


def run(env, runner, speeds, duration, threshold):
    """Roll out with fixed forward commands; return per-step logs."""
    speeds = torch.tensor(speeds, dtype=torch.float, device=env.device)
    weight = 9.81 * env._backend.link_mass.sum(dim=1, keepdim=True)
    contact, base_vx, reset = [], [], []
    for _ in range(int(duration / env.dt)):
        # Re-impose every step: a reset after a fall resamples commands.
        env.commands[:] = 0.0
        env.commands[:, 0] = speeds
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        load = env.contact_forces[:, env.feet_indices, 2] / weight
        contact.append((load > threshold).cpu().numpy())
        base_vx.append(env.base_lin_vel[:, 0].cpu().numpy())
        reset.append(env.terminated.cpu().numpy())
    return {
        "time": np.arange(1, len(contact) + 1) * env.dt,
        "contact": np.stack(contact, axis=1),  # [env, step, foot]
        "base_vx": np.stack(base_vx, axis=1),  # [env, step]
        "reset": np.stack(reset, axis=1),  # [env, step]
    }


def contact_intervals(time, in_contact, dt):
    """(start, width) of each contiguous contact block."""
    edges = np.diff(np.concatenate(([0], in_contact.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(time[s] - dt, (e - s) * dt) for s, e in zip(starts, ends)]


def plot(logs, speeds, leg_names, legs, window, dt, path):
    time = logs["time"]
    shown = time > time[-1] - window
    fig, axes = plt.subplots(
        len(speeds),
        1,
        figsize=(9, 1.0 + 0.45 * len(legs) * len(speeds)),
        sharex=True,
        squeeze=False,
    )
    for row, (ax, speed) in enumerate(zip(axes[:, 0], speeds)):
        for y, leg in enumerate(legs):
            blocks = contact_intervals(
                time[shown], logs["contact"][row, shown, leg], dt
            )
            ax.broken_barh(blocks, (y + 0.1, 0.8), facecolors="black")
        for t in time[shown][logs["reset"][row, shown]]:
            ax.axvline(t, color="tab:red", lw=1.0)
        mean_vx = logs["base_vx"][row, shown].mean()
        ax.set_title(
            f"commanded {speed:.2f} m/s, measured {mean_vx:.2f} m/s",
            fontsize=10,
            loc="left",
        )
        ax.set_yticks(np.arange(len(legs)) + 0.5)
        ax.set_yticklabels([leg_names[leg] for leg in legs])
        ax.set_ylim(0, len(legs))
        ax.invert_yaxis()
        ax.set_xlim(time[shown][0] - dt, time[-1])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved footfall diagram to {path}")


if __name__ == "__main__":
    args = get_args()
    play_argv = ["--task", args.task, "--headless", "--no-keyboard", "--no-record"]
    play_argv += ["--num_envs", str(len(args.speeds)), "--seed", str(args.seed)]
    play_argv += ["--checkpoint", str(args.checkpoint)]
    if args.experiment_name is not None:
        play_argv += ["--experiment_name", args.experiment_name]
    if args.load_run is not None:
        play_argv += ["--load_run", args.load_run]

    with torch.no_grad():
        env, runner, _ = play.setup(play.get_play_args(play_argv))
        if args.duration > env.max_episode_length_s:
            raise ValueError(
                f"--duration {args.duration} exceeds the "
                f"{env.max_episode_length_s} s episode length"
            )
        logs = run(env, runner, args.speeds, args.duration, args.contact_threshold)

    foot_names = env.robot_layout.body_groups["feet"]
    leg_names = [name.removesuffix("_foot") for name in foot_names]
    # Locked legs never step; show only the legs the policy controls.
    active = getattr(env, "active_legs", torch.ones(len(foot_names), dtype=bool))
    legs = [i for i, is_active in enumerate(active.tolist()) if is_active]

    path = Path(args.out or f"footfall_{args.task}.png")
    np.savez(
        path.with_suffix(".npz"),
        speeds=np.array(args.speeds),
        leg_names=np.array(leg_names),
        active_legs=np.array(active.tolist()),
        **logs,
    )
    plot(logs, args.speeds, leg_names, legs, args.window, env.dt, path)
