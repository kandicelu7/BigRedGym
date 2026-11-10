"""Sweep commanded speed and oscillator frequency and draw gait diagrams.

Runs a trained go2osc policy headless, one environment per
(disabled leg, oscillator frequency, commanded speed), and saves a gait
diagram per (disabled leg, frequency) plus a CSV of steady-state gait
statistics.

Usage:
    uv run --frozen scripts/gait_sweep.py --task go2osc \
        --speeds 0.75 1.25 2.5 --freqs 2.5
    uv run --frozen scripts/gait_sweep.py --task go2osc \
        --speeds 0.5 1.0 1.5 2.0 --freqs 1.75 2.5 3.25 4.0 --disable FR RL
"""

import argparse
import csv
import os
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gym.utils.helpers import get_load_path  # noqa: E402
from play import setup  # noqa: E402

LEG_LABELS = ["FL", "FR", "RL", "RR"]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", type=str, default="go2osc")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--backend", type=str, default="mujoco")
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[0.75, 1.25, 2.5],
        help="commanded forward speeds [m/s], one diagram panel each",
    )
    parser.add_argument(
        "--freqs",
        type=float,
        nargs="+",
        default=[2.5],
        help="intrinsic oscillator frequencies [Hz], one figure each",
    )
    parser.add_argument(
        "--disable",
        type=str,
        nargs="+",
        default=None,
        choices=LEG_LABELS + ["none"],
        help="disabled leg(s) to evaluate, one figure each (default: the "
        "task config's disabled leg). A policy trained with one leg "
        "disabled has not seen the others.",
    )
    parser.add_argument(
        "--duration", type=float, default=15.0, help="simulated time [s]"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="final stretch of the run that is plotted [s]",
    )
    parser.add_argument(
        "--stats_window",
        type=float,
        default=6.0,
        help="final stretch of the run used for the CSV statistics [s]",
    )
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=0.02,
        help="vertical foot force, as a fraction of body weight, counted as stance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="default: <run directory>/gait_sweep",
    )
    return parser.parse_args(argv)


def run_sweep(env, runner, grid, duration):
    """Roll the policy with pinned commands. Returns per-step logs (T, N, ...)."""
    env.disabled_leg_idx[:] = torch.tensor(
        [cell.disabled_idx for cell in grid], device=env.device
    )
    env.reset()
    speeds = torch.tensor([cell.speed for cell in grid], device=env.device)
    freqs = torch.tensor([cell.freq for cell in grid], device=env.device)
    weight = 9.81 * env._backend.link_mass.sum(dim=1, keepdim=True)

    n_steps = int(round(duration / env.dt))
    load = torch.zeros(n_steps, env.num_envs, env.num_legs)
    forward_vel = torch.zeros(n_steps, env.num_envs)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for step in range(n_steps):
        # Pinned every step so nothing in the task can resample them.
        env.commands[:, 0] = speeds
        env.commands[:, 1:] = 0.0
        env.phase_frequency[:, 0] = freqs
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        fell |= env.terminated
        load[step] = (env.contact_forces[:, env.feet_indices, 2] / weight).cpu()
        forward_vel[step] = env.base_lin_vel[:, 0].cpu()
    return load.numpy(), forward_vel.numpy(), fell.cpu().numpy()


def stance_intervals(contact, dt):
    """(start, length) of each run of True samples, in seconds from index 0."""
    padded = np.concatenate(([False], contact, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    return [(s * dt, (e - s) * dt) for s, e in zip(starts, ends)]


def touchdown_times(contact, dt, min_swing=0.04):
    """Stance onset times, ignoring breaks in contact shorter than min_swing."""
    times = []
    last_end = -np.inf
    for start, length in stance_intervals(contact, dt):
        if start - last_end >= min_swing:
            times.append(start)
        last_end = start + length
    return np.array(times)


def gait_statistics(contact, dt, active):
    """Duty factor, stride frequency and phase lags from a contact window.

    Phase lag is a leg's touchdown time after the reference leg's (the first
    active leg), as a fraction of the reference stride. ``lag_spread`` is the
    circular standard deviation of that lag in cycles: near 0 means the
    footfall pattern repeats, i.e. the gait has settled.
    """
    stats = {"duty": contact.mean(axis=0)}
    lag = np.full(contact.shape[1], np.nan)
    spread = np.full(contact.shape[1], np.nan)
    stats.update(stride_hz=np.nan, lag=lag, lag_spread=spread)

    reference = touchdown_times(contact[:, active[0]], dt)
    if len(reference) < 3:
        return stats
    stats["stride_hz"] = 1.0 / np.median(np.diff(reference))
    for leg in active:
        touchdowns = touchdown_times(contact[:, leg], dt)
        stride = np.searchsorted(reference, touchdowns, side="right") - 1
        valid = (stride >= 0) & (stride < len(reference) - 1)
        if not valid.any():
            continue
        start = reference[stride[valid]]
        period = reference[stride[valid] + 1] - start
        angle = 2 * np.pi * (touchdowns[valid] - start) / period
        mean_vector = np.exp(1j * angle).mean()
        lag[leg] = (np.angle(mean_vector) / (2 * np.pi)) % 1.0
        resultant = min(abs(mean_vector), 1.0)
        spread[leg] = np.sqrt(-2 * np.log(max(resultant, 1e-12))) / (2 * np.pi)
    return stats


def plot_gait_diagram(path, panels, dt, t_start, t_end, show_freq):
    """One panel per commanded speed; black bars are stance."""
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, 1.2 + 1.5 * len(panels)),
        sharex=True,
        squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels):
        active = panel["active"]
        for row, leg in enumerate(active):
            bars = [
                (t_start + start, length)
                for start, length in stance_intervals(panel["contact"][:, leg], dt)
            ]
            ax.broken_barh(bars, (row - 0.4, 0.8), facecolors="black")
        ax.set_yticks(range(len(active)), [LEG_LABELS[leg] for leg in active])
        ax.set_ylim(len(active) - 0.4, -0.6)
        ax.set_xlim(t_start, t_end)
        title = (
            f"commanded {panel['speed']:.2f} m/s, "
            f"measured {panel['measured_speed']:.2f} m/s"
        )
        if show_freq:
            title += f", oscillator {panel['freq']:.2f} Hz"
        if panel["fell"]:
            title += "  (fell during run)"
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    args = get_args()
    import gym.envs  # noqa: F401  (registers tasks)
    from gym.utils.task_registry import task_registry

    if args.disable is None:
        cfg_leg = task_registry.get_cfgs(args.task)[0].osc.disabled_leg
        args.disable = ["none" if cfg_leg is None else cfg_leg.split("_")[0]]
    grid = [
        SimpleNamespace(
            disabled=disabled,
            disabled_idx=-1 if disabled == "none" else LEG_LABELS.index(disabled),
            freq=freq,
            speed=speed,
        )
        for disabled in args.disable
        for freq in args.freqs
        for speed in args.speeds
    ]

    play_args = SimpleNamespace(
        task=args.task,
        device=args.device,
        num_envs=len(grid),
        seed=args.seed,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        original_cfg=args.original_cfg,
        headless=True,
        backend=args.backend,
        viewer_ui=False,
        no_track=True,
    )
    with torch.no_grad():
        env, runner = setup(play_args)
        load, forward_vel, fell = run_sweep(env, runner, grid, args.duration)

    dt = float(env.dt)
    contact = load > args.contact_threshold
    n_plot = int(round(args.window / dt))
    n_stats = int(round(min(args.stats_window, args.duration) / dt))

    output_dir = args.output_dir
    if output_dir is None:
        checkpoint_path = get_load_path(
            runner.cfg["experiment_name"],
            load_run=runner.cfg["load_run"],
            checkpoint=runner.cfg["checkpoint"],
        )
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "gait_sweep")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    panels = {}
    for env_id, cell in enumerate(grid):
        active = [leg for leg in range(len(LEG_LABELS)) if leg != cell.disabled_idx]
        stats = gait_statistics(contact[-n_stats:, env_id], dt, active)
        measured_speed = float(forward_vel[-n_stats:, env_id].mean())
        row = {
            "disabled_leg": cell.disabled,
            "oscillator_hz": cell.freq,
            "commanded_speed": cell.speed,
            "measured_speed": round(measured_speed, 4),
            "fell": bool(fell[env_id]),
            "stride_hz": round(float(stats["stride_hz"]), 4),
        }
        for name in ("duty", "lag", "lag_spread"):
            for leg, label in enumerate(LEG_LABELS):
                row[f"{name}_{label}"] = round(float(stats[name][leg]), 4)
        rows.append(row)
        panels.setdefault((cell.disabled, cell.freq), []).append(
            {
                "contact": contact[-n_plot:, env_id],
                "active": active,
                "speed": cell.speed,
                "freq": cell.freq,
                "measured_speed": measured_speed,
                "fell": bool(fell[env_id]),
            }
        )

    csv_path = os.path.join(output_dir, "gait_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    for (disabled, freq), figure_panels in panels.items():
        path = os.path.join(output_dir, f"gait_disabled-{disabled}_{freq:.2f}Hz.png")
        plot_gait_diagram(
            path,
            figure_panels,
            dt,
            args.duration - args.window,
            args.duration,
            show_freq=len(args.freqs) > 1,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
