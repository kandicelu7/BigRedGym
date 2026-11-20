"""Footfall diagrams of example gaits, one panel per (policy, speed, frequency).

Each example runs a batch of robots exactly as grid_eval.py does (random
initial phases, settle, then measure), classifies every robot the same way,
and plots the robot most typical of the batch's most common label: the one
whose touchdown timing is closest to the label group's circular mean. Stride
starts of the reference leg are drawn as thin lines, and each leg's measured
touchdown offset (fraction of the reference stride) is printed beside it, so
"irregular" robots can be read against the named gaits.

Each example runs in its own process (one env per task); results are cached
as .npz next to the figure, so --plot_only redraws without simulating.

Usage:
    uv run --frozen scripts/gait_examples.py            # default examples
    uv run --frozen scripts/gait_examples.py --plot_only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gym.utils.helpers import GYM_ROOT_DIR

# (task, run, speed [m/s], omega [Hz], why it is shown)
EXAMPLES = [
    ("go2osc_omega", "Nov10_08-26-28_", 1.5, 3.0, "trot"),
    ("go2osc_omega", "Nov10_07-40-07_", 1.5, 3.0, "trot, split diagonal"),
    ("go2osc_omega", "Nov10_09-29-25_", 2.0, 3.0, "pronk (FL barely loaded)"),
    ("go2osc_front_locked_omega", "Nov10_05-47-12_", 1.0, 3.0, "hop"),
    ("go2osc_front_locked_omega", "Nov10_07-55-19_", 1.5, 3.0, "rear-first hop"),
    ("go2osc_front_locked_omega", "Nov10_07-55-19_", 1.0, 1.5, "irregular: messy, repeated touchdowns"),
    ("go2osc_front_locked_omega_gait2", "Nov10_10-00-19_", 1.5, 3.0, "diagonal pair | RL"),
    ("go2osc_front_locked_omega_robust", "Nov10_10-15-20_", 2.5, 3.0, "rear-first hop, robust policy"),
    ("go2osc_hind_locked_omega", "Nov10_07-25-31_", 1.0, 3.0, "hop"),
    ("go2osc_hind_locked_omega", "Nov10_08-10-44_", 2.5, 3.0, "rear-first hop"),
]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Footfall diagrams of example gaits")
    parser.add_argument("--robots", type=int, default=32, help="robots per example")
    parser.add_argument("--window", type=float, default=1.5, help="plotted time [s]")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4, help="parallel simulations")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(GYM_ROOT_DIR) / "logs" / "sweep_results" / "gait_examples.png"),
    )
    # Internal: simulate one example (JSON) into an .npz.
    parser.add_argument("--one", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def cache_path(out, example):
    task, run, speed, omega, _ = example
    return Path(out).with_suffix("") / f"{task}__{run}__v{speed:g}_w{omega:g}.npz"


def typical_robot(env, log, args):
    """Classify every robot of a grid_eval.rollout log; return the most
    common label, its share, and the robot whose timing is closest to that
    label group's circular mean, with the leg bookkeeping used to classify."""
    import grid_eval

    feet = env.robot_layout.body_groups["feet"]
    legs = [k for k, on in enumerate(env.active_legs.tolist()) if on]
    names = [feet[k].removesuffix("_foot") for k in legs]
    ref, others, gaits = grid_eval.gait_reference(legs, names)
    robots_ok = np.flatnonzero(~log["fell"])
    timings, labels = [], []
    for robot in robots_ok:
        timing, _, _, label = grid_eval.robot_gait(
            log["contact"][robot], ref, others, legs, gaits, env.dt, args
        )
        timings.append(timing)
        labels.append(label)
    timings = np.array(timings)
    label = max(set(labels), key=labels.count)
    group = np.array([l == label for l in labels])
    centre = grid_eval.circular_mean(timings[group])
    gap = np.abs(timings[group] - centre) % 1.0
    distance = np.nan_to_num(np.linalg.norm(np.minimum(gap, 1 - gap), axis=1), nan=9.0)
    pick = np.flatnonzero(group)[distance.argmin()]
    return {"robot": robots_ok[pick], "label": label, "share": group.mean(),
            "timing": timings[pick], "legs": legs, "names": names, "ref": ref,
            "others": others}


def simulate(example, robots, path):
    """Run one example, pick its most typical robot, and save its contact."""
    import torch

    import grid_eval
    import play

    task, run, speed, omega, _ = example
    args = grid_eval.get_args(["--task", task, "--load_run", run, "--robots", str(robots)])
    play_argv = ["--task", task, "--load_run", run, "--headless", "--no-keyboard",
                 "--no-record", "--num_envs", str(robots), "--seed", "0"]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
        log = grid_eval.rollout(env, runner, speed, args, omega)

    pick = typical_robot(env, log, args)
    robot, label, legs, names, ref = (pick[k] for k in ("robot", "label", "legs", "names", "ref"))
    dt = env.dt

    np.savez(
        path,
        contact=log["contact"][robot][:, legs],
        names=np.array(names),
        ref=names[legs.index(ref)],
        others=np.array([names[legs.index(o)] for o in pick["others"]]),
        timing=pick["timing"],
        label=label,
        share=pick["share"],
        vx=log["vx"][robot],
        dt=dt,
        robot_label=grid_eval.robot_label(env.cfg.leg_lock),
        model=f"{Path(load_path).parent.name}/{Path(load_path).stem}",
    )
    print(f"Saved {path}")


def plot(examples, out, window):
    import footfall

    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 0.55 + 1.05 * n), squeeze=False)
    for ax, example in zip(axes[:, 0], examples):
        task, run, speed, omega, why = example
        d = np.load(cache_path(out, example), allow_pickle=False)
        contact, names, dt = d["contact"], list(d["names"]), float(d["dt"])
        steps = int(round(window / dt))
        contact = contact[-steps:]
        time = np.arange(1, steps + 1) * dt
        ref = names.index(str(d["ref"]))
        offsets = dict(zip(d["others"], d["timing"]))
        for y, name in enumerate(names):
            blocks = footfall.contact_intervals(time, contact[:, y], dt)
            ax.broken_barh(blocks, (y + 0.15, 0.7), facecolors="#3d3d3a")
        for t in footfall.touchdowns(contact[:, ref], int(round(0.03 / dt))):
            ax.axvline(time[t] - dt, color="#2a78d6", lw=0.8, zorder=0)
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_yticklabels(
            [f"{n} (ref)" if n == d["ref"] else f"{n} {offsets[n]:.2f}" for n in names],
            fontsize=8,
        )
        ax.set_ylim(0, len(names))
        ax.invert_yaxis()
        ax.set_xlim(0, window)
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            f"{d['robot_label']}, {run}: {speed:g} m/s, {omega:g} Hz.  "
            f"Classified '{d['label']}' ({float(d['share']):.0%} of robots)",
            fontsize=9, loc="left",
        )
    axes[-1, 0].set_xlabel("time [s] (last part of the measured window)")
    fig.suptitle(
        "Footfall of a typical robot per gait (dark: foot on ground; blue lines: "
        "reference-leg touchdowns;\nnumbers: when each leg lands, as a fraction of "
        "the reference stride after the reference leg)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main():
    args = get_args()
    if args.one is not None:
        simulate(tuple(json.loads(args.one)), args.robots, Path(args.out))
        return
    if not args.plot_only:
        todo = [e for e in EXAMPLES if not cache_path(args.out, e).exists()]
        cache_path(args.out, EXAMPLES[0]).parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(max(1, 16 // args.jobs)))
        running = []
        for example in todo:
            cmd = [sys.executable, __file__, "--one", json.dumps(example),
                   "--robots", str(args.robots), "--out", str(cache_path(args.out, example))]
            running.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL))
            if len(running) >= args.jobs:
                running.pop(0).wait()
        for p in running:
            p.wait()
    plot(EXAMPLES, args.out, args.window)


if __name__ == "__main__":
    main()
