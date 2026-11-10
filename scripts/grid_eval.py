"""Evaluate the (speed, frequency) sweep of front-locked policies as heatmaps.

For every sweep cell (go2osc_config.SWEEP_TASKS) with a trained run, runs that
cell's policy at its own commanded speed and oscillator frequency, from random
initial phases, and measures:

    speed tracking (measured / commanded), fall rate,
    resulting gait (footfall timing matched to named three-legged gaits),
    peak landing force (x body weight), and how evenly the legs carry load.

Results are plotted as heatmaps with speed on the x axis and frequency on the
y axis, and saved as .npz next to the figure.

Usage:
    uv run --frozen scripts/grid_eval.py
    uv run --frozen scripts/grid_eval.py --checkpoint 300 --out logs/sweep_results/it300.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import footfall
import play
from gym.envs.go2.go2osc_config import (
    SWEEP_OMEGAS,
    SWEEP_SPEEDS,
    sweep_task_name,
)
from gym.utils.helpers import GYM_ROOT_DIR


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the speed/frequency sweep")
    parser.add_argument(
        "--checkpoint", type=int, default=-1, help="model iteration (default: latest)"
    )
    parser.add_argument("--robots", type=int, default=64, help="robots per cell")
    parser.add_argument("--settle", type=float, default=10.0, help="settling time [s]")
    parser.add_argument("--measure", type=float, default=10.0, help="measured time [s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=0.05,
        help="vertical foot force, as a fraction of body weight, counted as contact",
    )
    parser.add_argument(
        "--landing_window",
        type=float,
        default=0.05,
        help="time after touchdown searched for the landing peak [s]",
    )
    parser.add_argument(
        "--gait_tolerance",
        type=float,
        default=0.18,
        help="max distance (fraction of a stride) to a named gait; farther is "
        "'irregular'. Half the smallest spacing between the named gaits.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(GYM_ROOT_DIR) / "logs" / "sweep_results" / "grid_eval.png"),
        help="figure path; results are saved next to it as .npz",
    )
    return parser.parse_args(argv)


def named_gaits(names):
    """Ideal gaits as touchdown times of legs a, b after the reference leg,
    in fractions of the reference leg's stride."""
    ref, a, b = names
    return [
        ("hop (all together)", (0.0, 0.0)),
        (f"{ref}+{b} | {a}", (0.5, 0.0)),
        (f"{ref}+{a} | {b}", (0.0, 0.5)),
        (f"{a}+{b} | {ref}", (0.5, 0.5)),
        (f"3-beat {ref}>{a}>{b}", (1 / 3, 2 / 3)),
        (f"3-beat {ref}>{b}>{a}", (2 / 3, 1 / 3)),
    ]


def classify(timing, gaits, tolerance):
    """Nearest named gait on the torus of touchdown timings, or 'irregular'."""
    if np.isnan(timing).any():
        return "not stepping"
    best, best_distance = None, np.inf
    for name, ideal in gaits:
        gap = np.abs(np.asarray(timing) - ideal)
        distance = np.linalg.norm(np.minimum(gap, 1.0 - gap))
        if distance < best_distance:
            best, best_distance = name, distance
    return best if best_distance <= tolerance else "irregular"


def rollout(env, runner, speed, args):
    """Run every robot at `speed`; returns per-step contact, peaks and speed."""
    weight = 9.81 * env._backend.link_mass.sum(dim=1, keepdim=True)
    sim_dt = env.dt / env.cfg.control.decimation
    peak = torch.zeros(env.num_envs, len(env.feet_indices))
    impulse = torch.zeros_like(peak)
    post_physics_step = env._post_physics_step

    # Peak and impulse of vertical foot force over every physics substep.
    def hooked_post_physics_step():
        post_physics_step()
        force = env.contact_forces[:, env.feet_indices, 2].clamp(min=0.0)
        torch.maximum(peak, force, out=peak)
        impulse.add_(force * sim_dt)

    env._post_physics_step = hooked_post_physics_step
    fell = torch.zeros(env.num_envs, dtype=torch.bool)
    contact, step_peak, vx = [], [], []
    n_settle = int(args.settle / env.dt)
    try:
        for i in range(int((args.settle + args.measure) / env.dt)):
            env.commands[:] = 0.0
            env.commands[:, 0] = speed
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            peak.zero_()
            if i == n_settle:
                impulse.zero_()
            env.step()
            fell |= env.terminated
            if i >= n_settle:
                load = env.contact_forces[:, env.feet_indices, 2] / weight
                contact.append((load > args.contact_threshold).numpy().copy())
                step_peak.append((peak / weight).numpy().copy())
                vx.append(env.base_lin_vel[:, 0].numpy().copy())
    finally:
        env._post_physics_step = post_physics_step
    return {
        "fell": fell.numpy(),
        "contact": np.stack(contact, axis=1),  # [robot, step, foot]
        "peak": np.stack(step_peak, axis=1),  # [robot, step, foot], x body weight
        "impulse": impulse.numpy().copy(),  # [robot, foot], measured window
        "vx": np.stack(vx, axis=1).mean(axis=1),  # [robot]
    }


def cell_metrics(log, speed, legs, names, dt, args):
    """Per-robot gait, landing peak and load shares, summarized for the cell."""
    ref, a, b = legs
    gaits = named_gaits(names)
    min_air = int(round(0.03 / dt))
    window = max(1, int(round(args.landing_window / dt)))
    ok = ~log["fell"]
    labels, timings, peaks = [], [], []
    for robot in np.flatnonzero(ok):
        contact = log["contact"][robot]
        _, fractions = footfall.relative_touchdowns(
            contact, ref, [a, b], min_air, int(round(2.0 / dt))
        )
        timing = []
        for leg in (a, b):
            f = fractions[leg][~np.isnan(fractions[leg])]
            # Circular mean of the touchdown time, as a fraction in [0, 1).
            mean = np.angle(np.exp(2j * np.pi * f).mean()) / (2 * np.pi) if len(f) >= 3 else np.nan
            timing.append(mean % 1.0)
        timings.append(timing)
        labels.append(classify(timing, gaits, args.gait_tolerance))
        landings = [
            log["peak"][robot, td : td + window, leg].max()
            for leg in legs
            for td in footfall.touchdowns(contact[:, leg], min_air)
        ]
        peaks.append(np.percentile(landings, 95) if landings else np.nan)

    shares = log["impulse"][ok][:, legs]
    shares = shares / shares.sum(axis=1, keepdims=True)
    names_list = [name for name, _ in gaits] + ["irregular", "not stepping"]
    counts = np.array([labels.count(name) for name in names_list])
    return {
        "speed_ratio": log["vx"][ok].mean() / speed if ok.any() else np.nan,
        "fall_rate": log["fell"].mean(),
        "gait_counts": counts,
        "timing": np.nanmean(np.array(timings), axis=0) if timings else np.full(2, np.nan),
        "peak_force": np.nanmean(peaks) if peaks else np.nan,
        "leg_shares": shares.mean(axis=0) if ok.any() else np.full(3, np.nan),
        "unevenness": (shares.std(axis=1) / shares.mean(axis=1)).mean() if ok.any() else np.nan,
    }, names_list


def annotate(ax, grid, fmt):
    for (row, col), value in np.ndenumerate(grid):
        if not np.isnan(value):
            ax.text(col, row, fmt(value), ha="center", va="center", fontsize=8, color="white")


def plot(results, gait_names, leg_names, path, title):
    speeds, omegas = np.array(SWEEP_SPEEDS), np.array(SWEEP_OMEGAS)
    shape = (len(omegas), len(speeds))

    def grid(key, reduce=lambda v: v):
        out = np.full(shape, np.nan)
        for (i, j), cell in results.items():
            out[i, j] = reduce(cell[key])
        return out

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("speed_ratio", "measured / commanded speed", "RdBu", (0, 2), "{:.2f}"),
        ("fall_rate", "fall rate", "Reds", (0, 1), "{:.0%}"),
        (None, "dominant gait (share of robots)", None, None, None),
        ("peak_force", "peak landing force [x body weight]", "plasma", None, "{:.2f}"),
        ("unevenness", "uneven load across legs [CV, 0 = even]", "cividis", None, "{:.2f}"),
        ("leg_shares", "load share per leg", "Greys", (1 / 3, 1.0), None),
    ]
    for ax, (key, label, cmap, limits, fmt) in zip(axes.ravel(), panels):
        if key is None:  # categorical gait map
            dominant = grid("gait_counts", lambda c: np.argmax(c) if c.sum() else np.nan)
            share = grid("gait_counts", lambda c: c.max() / c.sum() if c.sum() else np.nan)
            colors = plt.cm.tab10(np.arange(len(gait_names)))
            ax.imshow(dominant, cmap=ListedColormap(colors), vmin=-0.5,
                      vmax=len(gait_names) - 0.5, origin="lower", aspect="auto")
            for (row, col), value in np.ndenumerate(dominant):
                if not np.isnan(value):
                    ax.text(col, row, f"{gait_names[int(value)]}\n{share[row, col]:.0%}",
                            ha="center", va="center", fontsize=7, color="white")
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in range(len(gait_names))]
            ax.legend(handles, gait_names, fontsize=6, loc="upper left",
                      bbox_to_anchor=(1.0, 1.0), frameon=False)
        elif key == "leg_shares":
            shares = grid("leg_shares", lambda s: np.nanmax(s))
            ax.imshow(shares, cmap=cmap, vmin=limits[0], vmax=limits[1],
                      origin="lower", aspect="auto")
            for (i, j), cell in results.items():
                s = cell["leg_shares"]
                if not np.isnan(s).any():
                    text = "\n".join(f"{n} {v:.0%}" for n, v in zip(leg_names, s))
                    ax.text(j, i, text, ha="center", va="center", fontsize=7, color="tab:red")
        else:
            values = grid(key)
            vmin, vmax = limits or (np.nanmin(values), np.nanmax(values))
            image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                              origin="lower", aspect="auto")
            fig.colorbar(image, ax=ax, shrink=0.85)
            annotate(ax, values, fmt.format)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
        ax.set_yticks(range(len(omegas)), [f"{w:g}" for w in omegas])
        ax.set_xlabel("commanded speed [m/s]")
        ax.set_ylabel("oscillator frequency [Hz]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved sweep heatmaps to {path}")


if __name__ == "__main__":
    args = get_args()
    results, gait_names, leg_names, sim_hz = {}, None, None, None
    for i, omega in enumerate(SWEEP_OMEGAS):
        for j, speed in enumerate(SWEEP_SPEEDS):
            task = sweep_task_name(speed, omega)
            if not (Path(GYM_ROOT_DIR) / "logs" / task).is_dir():
                print(f"{task}: no trained run, skipped")
                continue
            play_argv = ["--task", task, "--headless", "--no-keyboard", "--no-record"]
            play_argv += ["--num_envs", str(args.robots), "--seed", str(args.seed)]
            play_argv += ["--checkpoint", str(args.checkpoint)]
            with torch.no_grad():
                env, runner, load_path = play.setup(play.get_play_args(play_argv))
                log = rollout(env, runner, speed, args)
            legs = [k for k, active in enumerate(env.active_legs.tolist()) if active]
            feet = env.robot_layout.body_groups["feet"]
            leg_names = [feet[k].removesuffix("_foot") for k in legs]
            sim_hz = 1.0 / (env.dt / env.cfg.control.decimation)
            cell, gait_names = cell_metrics(log, speed, legs, leg_names, env.dt, args)
            cell["run"] = f"{Path(load_path).parent.name}/{Path(load_path).stem}"
            results[(i, j)] = cell
            counts = cell["gait_counts"]
            gait = (
                f"{gait_names[int(np.argmax(counts))]} {counts.max() / counts.sum():.0%}"
                if counts.sum()
                else "no robot stayed up"
            )
            print(
                f"{task}: speed x{cell['speed_ratio']:.2f}, falls {cell['fall_rate']:.0%}, "
                f"{gait}, peak {cell['peak_force']:.2f} BW, "
                f"unevenness {cell['unevenness']:.2f}"
            )
    if not results:
        raise SystemExit("No trained sweep cells found under logs/")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["speed_ratio", "fall_rate", "peak_force", "unevenness"]
    shape = (len(SWEEP_OMEGAS), len(SWEEP_SPEEDS))
    saved = {key: np.full(shape, np.nan) for key in keys}
    saved["gait_counts"] = np.zeros(shape + (len(gait_names),))
    saved["timing"] = np.full(shape + (2,), np.nan)
    saved["leg_shares"] = np.full(shape + (3,), np.nan)
    runs = np.full(shape, "", dtype=object)
    for (i, j), cell in results.items():
        for key in saved:
            saved[key][i, j] = cell[key]
        runs[i, j] = cell["run"]
    np.savez(
        path.with_suffix(".npz"),
        speeds=np.array(SWEEP_SPEEDS),
        omegas=np.array(SWEEP_OMEGAS),
        gait_names=np.array(gait_names),
        leg_names=np.array(leg_names),
        runs=runs.astype(str),
        **saved,
    )
    checkpoint = "latest" if args.checkpoint == -1 else f"model_{args.checkpoint}"
    title = (
        f"Front-right locked, one policy per cell ({checkpoint}); {args.robots} robots "
        f"per cell from random phases, measured {args.measure:g} s after "
        f"{args.settle:g} s; physics {sim_hz:.0f} Hz"
    )
    plot(results, gait_names, leg_names, path, title)
