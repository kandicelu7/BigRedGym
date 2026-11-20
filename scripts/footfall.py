"""Footfall (gait) diagram of a trained policy at several forward speeds.

Runs one headless env per commanded speed, records which feet are loaded each
control step, and plots time on the x axis against the unlocked legs on the
y axis: black where the foot is on the ground, empty otherwise.

With --ramp, instead runs several envs through one slow speed ramp (up, then
back down) and plots, over time, the commanded / measured speed, each leg's
touchdown relative to a reference leg's stride, and each leg's duty factor,
so gait transitions (and hysteresis between up and down) show up as jumps.

Usage:
    uv run --frozen scripts/footfall.py --task go2osc_front_locked
    uv run --frozen scripts/footfall.py --task go2osc --speeds 0.5 1 2 3 \
        --load_run Oct30_21-51-01_ --checkpoint 500 --out trot.png
    uv run --frozen scripts/footfall.py --task go2osc_front_locked --ramp 0.3 3
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
        "--original_cfg",
        action="store_true",
        help="use the configs saved with the selected run instead of the "
        "current ones (needed when the config changed after training)",
    )
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
    parser.add_argument(
        "--min_air",
        type=float,
        default=0.03,
        help="ramp mode: a touchdown only counts after at least this long in "
        "the air [s], so contact chatter is not read as extra steps",
    )
    parser.add_argument(
        "--ramp",
        type=float,
        nargs=2,
        default=None,
        metavar=("V_START", "V_END"),
        help="ramp mode: sweep the command from V_START to V_END and back [m/s]",
    )
    parser.add_argument(
        "--ramp_time",
        type=float,
        default=30.0,
        help="duration of each leg of the ramp [s]",
    )
    parser.add_argument(
        "--ramp_settle",
        type=float,
        default=5.0,
        help="time held at V_START before the ramp starts [s]",
    )
    parser.add_argument(
        "--ramp_envs", type=int, default=8, help="robots run through the ramp"
    )
    parser.add_argument("--out", type=str, default=None, help="figure path")
    return parser.parse_args(argv)


def run(env, runner, command, duration, threshold):
    """Roll out with forward command `command(t)` [m/s]; return per-step logs."""
    weight = 9.81 * env._backend.link_mass.sum(dim=1, keepdim=True)
    contact, base_vx, reset, omega, commanded = [], [], [], [], []
    for i in range(int(duration / env.dt)):
        # Re-impose every step: a reset after a fall resamples commands.
        env.commands[:] = 0.0
        env.commands[:, 0] = command(i * env.dt)
        # .cpu().numpy() on a CPU tensor is a view of the live env buffer;
        # copy, or every logged step would show the final value.
        commanded.append(env.commands[:, 0].cpu().numpy().copy())
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        load = env.contact_forces[:, env.feet_indices, 2] / weight
        contact.append((load > threshold).cpu().numpy())
        base_vx.append(env.base_lin_vel[:, 0].cpu().numpy().copy())
        reset.append(env.terminated.cpu().numpy().copy())
        # Oscillator frequency [Hz], for tasks with phase oscillators.
        if hasattr(env, "osc_omega"):
            omega.append(env.osc_omega[:, 0].cpu().numpy().copy())
    logs = {
        "time": np.arange(1, len(contact) + 1) * env.dt,
        "contact": np.stack(contact, axis=1),  # [env, step, foot]
        "base_vx": np.stack(base_vx, axis=1),  # [env, step]
        "reset": np.stack(reset, axis=1),  # [env, step]
        "command": np.stack(commanded, axis=1),  # [env, step]
    }
    if omega:
        logs["omega"] = np.stack(omega, axis=1)  # [env, step]
    return logs


def contact_intervals(time, in_contact, dt):
    """(start, width) of each contiguous contact block."""
    edges = np.diff(np.concatenate(([0], in_contact.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(time[s] - dt, (e - s) * dt) for s, e in zip(starts, ends)]


def drop_taps(contact, min_steps):
    """Contact [step, foot] without contact blocks shorter than `min_steps`:
    brief taps before a landing, which would otherwise count as touchdowns."""
    contact = np.asarray(contact, dtype=bool).copy()
    for foot in range(contact.shape[1]):
        edges = np.diff(np.concatenate(([0], contact[:, foot].astype(int), [0])))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start < min_steps:
                contact[start:end, foot] = False
    return contact


def touchdowns(in_contact, min_air_steps):
    """Steps where a foot lands after at least `min_air_steps` in the air."""
    on = in_contact.astype(bool)
    rising = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    on_steps = np.flatnonzero(on)
    previous_on = np.searchsorted(on_steps, rising) - 1
    air = np.where(
        previous_on >= 0, rising - on_steps[np.maximum(previous_on, 0)] - 1, rising
    )
    return rising[air >= min_air_steps]


def liftoffs(in_contact, min_air_steps):
    """Steps where a foot leaves the ground (its first airborne step) for at
    least `min_air_steps`."""
    on = in_contact.astype(bool)
    falling = np.flatnonzero(on[:-1] & ~on[1:]) + 1
    on_steps = np.flatnonzero(on)
    following = np.searchsorted(on_steps, falling)
    air = np.where(
        following < len(on_steps),
        on_steps[np.minimum(following, len(on_steps) - 1)] - falling,
        len(on) - falling,
    )
    return falling[air >= min_air_steps]


def relative_liftoffs(contact, ref, legs, min_air_steps, max_stride_steps):
    """Per stride of leg `ref` (liftoff to liftoff): when each leg lifts off,
    as a fraction of the stride, wrapped to [-0.25, 0.75) as in
    relative_touchdowns. Returns {leg: fractions}."""
    events = {leg: liftoffs(contact[:, leg], min_air_steps) for leg in legs}
    ref_events = liftoffs(contact[:, ref], min_air_steps)
    fractions = {leg: [] for leg in legs}
    for a, b in zip(ref_events[:-1], ref_events[1:]):
        if b - a > max_stride_steps:
            continue
        for leg in legs:
            inside = events[leg][(events[leg] >= a) & (events[leg] < b)]
            fraction = (inside[0] - a) / (b - a) if inside.size else np.nan
            fractions[leg].append((fraction + 0.25) % 1.0 - 0.25)
    return {k: np.array(v) for k, v in fractions.items()}


def relative_touchdowns(contact, ref, legs, min_air_steps, max_stride_steps):
    """Per stride of leg `ref`: when each leg lands, as a fraction of the stride.

    Returns the stride start steps and {leg: fractions}, wrapped to
    [-0.25, 0.75) so in-phase (0) and opposite (0.5) both read unsplit.
    """
    landings = {leg: touchdowns(contact[:, leg], min_air_steps) for leg in legs}
    ref_landings = touchdowns(contact[:, ref], min_air_steps)
    starts, fractions = [], {leg: [] for leg in legs}
    for a, b in zip(ref_landings[:-1], ref_landings[1:]):
        if b - a > max_stride_steps:  # not stepping, e.g. standing still
            continue
        starts.append(a)
        for leg in legs:
            inside = landings[leg][(landings[leg] >= a) & (landings[leg] < b)]
            fraction = (inside[0] - a) / (b - a) if inside.size else np.nan
            fractions[leg].append((fraction + 0.25) % 1.0 - 0.25)
    return np.array(starts, dtype=int), {k: np.array(v) for k, v in fractions.items()}


def moving_average(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def plot_ramp(logs, leg_names, legs, dt, min_air, path):
    time = logs["time"]
    one_second = max(1, int(round(1.0 / dt)))
    fell = logs["reset"].any(axis=1)
    fig, (ax_v, ax_p, ax_d) = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 1]},
    )

    ax_v.plot(time, logs["command"][0], "k--", lw=1, label="commanded")
    ax_v.plot(
        time,
        moving_average(logs["base_vx"][~fell].mean(axis=0), one_second),
        color="tab:blue",
        label="measured (mean, 1 s)",
    )
    ax_v.set_ylabel("speed [m/s]")
    ax_v.legend(loc="upper left", fontsize=8, frameon=False)
    if "omega" in logs:
        ax_w = ax_v.twinx()
        ax_w.plot(time, logs["omega"][0], color="tab:gray", lw=1)
        ax_w.set_ylabel("oscillator [Hz]", color="tab:gray")
        ax_w.tick_params(axis="y", colors="tab:gray")
    ax_v.set_title(
        f"{len(fell)} robots from random initial phases"
        + (f"; {fell.sum()} fell and are excluded" if fell.any() else ""),
        fontsize=10,
        loc="left",
    )

    ref, others = legs[0], legs[1:]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, leg in enumerate(others):
        label = leg_names[leg]
        for env_id in np.flatnonzero(~fell):
            starts, fractions = relative_touchdowns(
                logs["contact"][env_id],
                ref,
                others,
                int(round(min_air / dt)),
                one_second * 2,
            )
            ax_p.scatter(
                time[starts],
                fractions[leg],
                s=4,
                color=colors[k],
                alpha=0.5,
                label=label,
            )
            label = None
    for level in (0.0, 0.5):
        ax_p.axhline(level, color="black", lw=0.5, ls=":")
    ax_p.set_ylim(-0.25, 0.75)
    ax_p.set_ylabel(f"touchdown after {leg_names[ref]}\n[fraction of its stride]")
    ax_p.legend(loc="upper left", fontsize=8, frameon=False, markerscale=3)

    for k, leg in enumerate(legs):
        duty = logs["contact"][~fell][:, :, leg].mean(axis=0)
        ax_d.plot(time, moving_average(duty, one_second), label=leg_names[leg])
    ax_d.set_ylim(0, 1)
    ax_d.set_ylabel("duty factor")
    ax_d.set_xlabel("time [s]")
    ax_d.legend(loc="upper left", fontsize=8, frameon=False, ncol=len(legs))

    for ax in (ax_v, ax_p, ax_d):
        ax.set_xlim(time[0], time[-1])
        for side in ("top",):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved speed-ramp gait plot to {path}")


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
        title = f"commanded {speed:.2f} m/s, measured {mean_vx:.2f} m/s"
        if "omega" in logs:
            title += f", oscillator {logs['omega'][row, shown].mean():.2f} Hz"
        ax.set_title(
            title,
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
    num_envs = args.ramp_envs if args.ramp else len(args.speeds)
    play_argv = ["--task", args.task, "--headless", "--no-keyboard", "--no-record"]
    play_argv += ["--num_envs", str(num_envs), "--seed", str(args.seed)]
    play_argv += ["--checkpoint", str(args.checkpoint)]
    if args.experiment_name is not None:
        play_argv += ["--experiment_name", args.experiment_name]
    if args.load_run is not None:
        play_argv += ["--load_run", args.load_run]
    if args.original_cfg:
        play_argv += ["--original_cfg"]

    with torch.no_grad():
        env, runner, _ = play.setup(play.get_play_args(play_argv))
        if args.ramp:
            v_start, v_end = args.ramp
            duration = args.ramp_settle + 2 * args.ramp_time
            # One ramp is longer than play.py's episode; don't time out mid-ramp.
            env.max_episode_length = int(duration / env.dt) + 1

            def command(t):
                s = np.clip((t - args.ramp_settle) / args.ramp_time, 0.0, 2.0)
                return v_start + (v_end - v_start) * (s if s <= 1.0 else 2.0 - s)

        else:
            duration = args.duration
            if duration > env.max_episode_length_s:
                raise ValueError(
                    f"--duration {duration} exceeds the "
                    f"{env.max_episode_length_s} s episode length"
                )
            speeds = torch.tensor(args.speeds, dtype=torch.float, device=env.device)

            def command(t):
                return speeds

        logs = run(env, runner, command, duration, args.contact_threshold)

    foot_names = env.robot_layout.body_groups["feet"]
    leg_names = [name.removesuffix("_foot") for name in foot_names]
    # Locked legs never step; show only the legs the policy controls.
    active = getattr(env, "active_legs", torch.ones(len(foot_names), dtype=bool))
    legs = [i for i, is_active in enumerate(active.tolist()) if is_active]

    suffix = "_ramp" if args.ramp else ""
    path = Path(args.out or f"footfall_{args.task}{suffix}.png")
    np.savez(
        path.with_suffix(".npz"),
        speeds=np.array(args.ramp or args.speeds),
        leg_names=np.array(leg_names),
        active_legs=np.array(active.tolist()),
        **logs,
    )
    if args.ramp:
        plot_ramp(logs, leg_names, legs, env.dt, args.min_air, path)
    else:
        plot(logs, args.speeds, leg_names, legs, args.window, env.dt, path)
