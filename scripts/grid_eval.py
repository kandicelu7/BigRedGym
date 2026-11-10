"""Evaluate the (speed, frequency) sweep of front-locked policies as heatmaps.

For every sweep cell (go2osc_config.SWEEP_TASKS) with a trained run, runs that
cell's policy at its own commanded speed and oscillator frequency, from random
initial phases. With --task, runs that one policy (trained with
osc.omega_range) at every cell instead, forcing the cell's frequency. Measures:

    speed tracking (measured / commanded), fall rate,
    resulting gait (footfall timing matched to named three-legged gaits),
    peak landing force (x body weight), and how evenly the legs carry load.

Results are plotted as heatmaps with speed on the x axis and frequency on the
y axis, and saved as .npz next to the figure.

Usage:
    uv run --frozen scripts/grid_eval.py
    uv run --frozen scripts/grid_eval.py --checkpoint 300 --out logs/sweep_results/it300.png
    uv run --frozen scripts/grid_eval.py --task go2osc_front_locked_omega
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
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
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="one policy evaluated at every cell (default: one policy per cell)",
    )
    parser.add_argument(
        "--load_run", type=str, default=None, help="run of --task (default: latest)"
    )
    parser.add_argument("--robots", type=int, default=64, help="robots per cell")
    parser.add_argument("--settle", type=float, default=10.0, help="settling time [s]")
    parser.add_argument("--measure", type=float, default=10.0, help="measured time [s]")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.5,
        help="gait-over-time figure: classify every this many seconds [s]",
    )
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
        default=None,
        help="figure path; results are saved next to it as .npz (default: "
        "logs/sweep_results/grid_eval.png, or <task>.png with --task)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        name = args.task or "grid_eval"
        args.out = str(Path(GYM_ROOT_DIR) / "logs" / "sweep_results" / f"{name}.png")
    return args


def named_gaits(names):
    """Ideal gaits as touchdown times of legs a, b after the reference leg,
    in fractions of the reference leg's stride."""
    ref, a, b = names
    return [
        ("hop", (0.0, 0.0)),  # all three legs together
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


def rollout(env, runner, speed, args, omega=None):
    """Run every robot at `speed` (and oscillator frequency `omega`, for a
    policy trained with osc.omega_range); returns per-step contact, peaks and
    speed."""
    if omega is not None:
        # Fresh episodes, random phases, at the cell's frequency. Robots that
        # fall and reset draw a new frequency, so it is forced every step.
        env.reset()
        env.episode_omega[:] = omega
        env.osc_omega[:] = omega
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
    contact, step_peak, vx, contact_all = [], [], [], []
    n_settle = int(args.settle / env.dt)
    n_steps = int((args.settle + args.measure) / env.dt)
    fell_at = np.full(env.num_envs, n_steps)
    try:
        for i in range(n_steps):
            env.commands[:] = 0.0
            env.commands[:, 0] = speed
            if omega is not None:
                env.episode_omega[:] = omega
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
            fell_at[env.terminated.numpy() & (fell_at == n_steps)] = i
            load = env.contact_forces[:, env.feet_indices, 2] / weight
            contact_all.append((load > args.contact_threshold).numpy().copy())
            if i >= n_settle:
                contact.append(contact_all[-1])
                step_peak.append((peak / weight).numpy().copy())
                vx.append(env.base_lin_vel[:, 0].numpy().copy())
    finally:
        env._post_physics_step = post_physics_step
    return {
        "fell": fell.numpy(),
        "fell_at": fell_at,  # [robot], first step fallen (n_steps if never)
        "contact_all": np.stack(contact_all, axis=1),  # [robot, step, foot], whole run
        "contact": np.stack(contact, axis=1),  # [robot, step, foot]
        "peak": np.stack(step_peak, axis=1),  # [robot, step, foot], x body weight
        "impulse": impulse.numpy().copy(),  # [robot, foot], measured window
        "vx": np.stack(vx, axis=1).mean(axis=1),  # [robot]
    }


def circular_mean(fractions):
    """Mean over robots of stride fractions, wrapped (0.95 and 0.05 average to 0)."""
    angles = np.exp(2j * np.pi * fractions)
    valid = ~np.isnan(fractions)
    mean = np.where(valid, angles, 0).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    return np.where(valid.any(axis=0), np.angle(mean) / (2 * np.pi) % 1.0, np.nan)


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
        "timing": circular_mean(np.array(timings)) if timings else np.full(2, np.nan),
        "peak_force": np.nanmean(peaks) if peaks else np.nan,
        "leg_shares": shares.mean(axis=0) if ok.any() else np.full(3, np.nan),
        "unevenness": (shares.std(axis=1) / shares.mean(axis=1)).mean() if ok.any() else np.nan,
    }, names_list


def gait_shares_over_time(log, legs, names, gait_names, dt, args):
    """Share of robots in each gait every `args.interval` s over the whole run,
    as in Zhang et al. Fig. 5: RPD averaged over each robot's last two strides,
    nearest named gait within `gait_tolerance`, else irregular."""
    from gait_over_time import classify, last_strides_rpd  # avoids an import cycle

    gaits = [(n, tuple(2 * np.pi * np.asarray(r))) for n, r in named_gaits(names)]
    labels = gait_names + ["fell"]
    min_air = int(round(0.03 / dt))
    max_stride = int(round(2.0 / dt))
    times = np.arange(args.interval, args.settle + args.measure + 1e-9, args.interval)
    shares = np.zeros((len(times), len(labels)))
    contact = log["contact_all"]
    for t, time in enumerate(times):
        end = min(int(round(time / dt)), contact.shape[1])
        for robot in range(contact.shape[0]):
            if log["fell_at"][robot] < end:
                label = "fell"
            else:
                rpd = last_strides_rpd(
                    contact[robot, :end], legs[0], legs[1:], min_air, max_stride, 2
                )
                label = classify(rpd, gaits, 2 * np.pi * args.gait_tolerance)
                label = "irregular" if label == "transition" else label
            shares[t, labels.index(label)] += 1
    return times, shares / contact.shape[0], labels


# Reference categorical palette (fixed order) for the named gaits; greys for
# irregular, not stepping and fell, which are not gaits.
GAIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NON_GAIT_COLORS = {"irregular": "#c8c7c0", "not stepping": "#8a8980", "fell": "#3d3d3a"}


def gait_colors(labels):
    named = [label for label in labels if label not in NON_GAIT_COLORS]
    colors = dict(zip(named, GAIT_COLORS))
    colors.update(NON_GAIT_COLORS)
    return [colors[label] for label in labels]


def plot_over_time(results, labels, path, title, settle):
    """Small multiples: gait share over time per (speed, frequency) cell."""
    speeds, omegas = SWEEP_SPEEDS, SWEEP_OMEGAS
    fig, axes = plt.subplots(
        len(omegas), len(speeds), figsize=(2.6 * len(speeds) + 2.2, 1.9 * len(omegas)),
        sharex=True, sharey=True, squeeze=False,
    )
    colors = gait_colors(labels)
    used = np.zeros(len(labels), dtype=bool)
    for (i, j), cell in results.items():
        used |= cell["time_shares"].max(axis=0) > 0
    for row, i in enumerate(reversed(range(len(omegas)))):  # high frequency on top
        for j in range(len(speeds)):
            ax = axes[row, j]
            cell = results.get((i, j))
            if cell is None:
                ax.text(0.5, 0.5, "not trained", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#6b6a64")
            else:
                times, shares = cell["times"], cell["time_shares"]
                ax.stackplot(times, shares[:, used].T,
                             colors=[c for c, u in zip(colors, used) if u],
                             edgecolor="white", linewidth=0.6)
                ax.set_xlim(times[0], times[-1])
                ax.axvline(settle, color="#3d3d3a", lw=0.8, ls="--")
            ax.set_ylim(0, 1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=7, colors="#6b6a64")
            if row == 0:
                ax.set_title(f"{speeds[j]:g} m/s", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{omegas[i]:g} Hz\nshare", fontsize=8)
            if row == len(omegas) - 1:
                ax.set_xlabel("time [s]", fontsize=8)
    axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, u in zip(colors, used) if u]
    fig.legend(handles, [l for l, u in zip(labels, used) if u], loc="center right",
               fontsize=8, frameon=False, title="gait", title_fontsize=8)
    fig.suptitle(title + f"\ncolumns: commanded speed, rows: oscillator frequency; "
                 f"dashed line: end of settling ({settle:g} s)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1 - 2.0 / fig.get_figwidth(), 1))
    fig.savefig(path, dpi=130)
    print(f"Saved gait-over-time grid to {path}")


def readable_text_color(rgb):
    """Black on light backgrounds, white on dark ones."""
    r, g, b = rgb[:3]
    return "black" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.4 else "white"


def annotate(ax, grid, fmt, color="white", image=None):
    """Cell values; with `image`, black or white per cell for contrast."""
    for (row, col), value in np.ndenumerate(grid):
        if not np.isnan(value):
            if image is not None:
                color = readable_text_color(image.to_rgba(value))
            ax.text(col, row, fmt(value), ha="center", va="center", fontsize=8, color=color)


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
        # Ranges fitted to the data (see below) so small differences show.
        ("speed_ratio", "measured / commanded speed", "RdBu", None, "{:.2f}"),
        ("fall_rate", "fall rate", "Reds", None, "{:.0%}"),
        (None, "dominant gait (share of robots)", None, None, None),
        ("peak_force", "peak landing force [x body weight]", "plasma", None, "{:.2f}"),
        ("unevenness", "uneven load across legs [CV, 0 = even]", "cividis", None, "{:.2f}"),
        ("leg_shares", "load share per leg", "Greys", (1 / 3, 1.0), None),
    ]
    for ax, (key, label, cmap, limits, fmt) in zip(axes.ravel(), panels):
        if key is None:  # categorical gait map
            dominant = grid("gait_counts", lambda c: np.argmax(c) if c.sum() else np.nan)
            share = grid("gait_counts", lambda c: c.max() / c.sum() if c.sum() else np.nan)
            colors = gait_colors(gait_names)
            ax.imshow(dominant, cmap=ListedColormap(colors), vmin=-0.5,
                      vmax=len(gait_names) - 0.5, origin="lower", aspect="auto")
            for (row, col), value in np.ndenumerate(dominant):
                if not np.isnan(value):
                    rgb = matplotlib.colors.to_rgb(colors[int(value)])
                    light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.5
                    ax.text(col, row, f"{gait_names[int(value)]}\n{share[row, col]:.0%}",
                            ha="center", va="center", fontsize=7,
                            color="#1a1a19" if light else "white")
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
            contrast = key in ("speed_ratio", "fall_rate")
            if key == "speed_ratio":
                # Diverging around perfect tracking (1.0), symmetric so equal
                # over- and undershoot get equal color strength.
                spread = max(np.nanmax(np.abs(values - 1.0)), 0.05)
                vmin, vmax = 1.0 - spread, 1.0 + spread
            elif key == "fall_rate":
                vmin, vmax = 0.0, max(np.nanmax(values), 0.05)
            image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                              origin="lower", aspect="auto")
            bar = fig.colorbar(image, ax=ax, shrink=0.85)
            if key == "fall_rate":
                bar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            annotate(ax, values, fmt.format, image=image if contrast else None)
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

    def load(task):
        play_argv = ["--task", task, "--headless", "--no-keyboard", "--no-record"]
        play_argv += ["--num_envs", str(args.robots), "--seed", str(args.seed)]
        play_argv += ["--checkpoint", str(args.checkpoint)]
        if args.load_run is not None:
            play_argv += ["--load_run", args.load_run]
        with torch.no_grad():
            return play.setup(play.get_play_args(play_argv))

    if args.task is not None:
        env, runner, load_path = load(args.task)
        if env.cfg.osc.omega_range is None:
            raise SystemExit(f"{args.task} was not trained with osc.omega_range")
    for i, omega in enumerate(SWEEP_OMEGAS):
        for j, speed in enumerate(SWEEP_SPEEDS):
            if args.task is None:
                task = sweep_task_name(speed, omega)
                if not (Path(GYM_ROOT_DIR) / "logs" / task).is_dir():
                    print(f"{task}: no trained run, skipped")
                    continue
                env, runner, load_path = load(task)
                with torch.no_grad():
                    log = rollout(env, runner, speed, args)
            else:
                task = f"{args.task} v{speed:g} w{omega:g}"
                with torch.no_grad():
                    log = rollout(env, runner, speed, args, omega=omega)
            legs = [k for k, active in enumerate(env.active_legs.tolist()) if active]
            feet = env.robot_layout.body_groups["feet"]
            leg_names = [feet[k].removesuffix("_foot") for k in legs]
            sim_hz = 1.0 / (env.dt / env.cfg.control.decimation)
            cell, gait_names = cell_metrics(log, speed, legs, leg_names, env.dt, args)
            cell["times"], cell["time_shares"], time_labels = gait_shares_over_time(
                log, legs, leg_names, gait_names, env.dt, args
            )
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
    times = next(iter(results.values()))["times"]
    saved["time_shares"] = np.full(shape + (len(times), len(time_labels)), np.nan)
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
        times=times,
        time_labels=np.array(time_labels),
        leg_names=np.array(leg_names),
        runs=runs.astype(str),
        **saved,
    )
    checkpoint = "latest" if args.checkpoint == -1 else f"model_{args.checkpoint}"
    title = (
        f"Front-right locked, "
        + (f"one policy for all cells ({runs[0, 0]})" if args.task
           else f"one policy per cell ({checkpoint})")
        + f"; {args.robots} robots "
        f"per cell from random phases, measured {args.measure:g} s after "
        f"{args.settle:g} s; physics {sim_hz:.0f} Hz"
    )
    plot(results, gait_names, leg_names, path, title)
    plot_over_time(results, time_labels,
                   path.with_name(path.stem + "_over_time.png"), title, args.settle)
