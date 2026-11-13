"""Evaluate the (speed, frequency) sweep of front-locked policies as heatmaps.

For every sweep cell (go2osc_config.SWEEP_TASKS) with a trained run, runs that
cell's policy at its own commanded speed and oscillator frequency, from random
initial phases. With --task, runs that one policy (trained with
osc.omega_range) at every cell instead, forcing the cell's frequency. Measures:

    speed tracking (measured / commanded), fall rate,
    resulting gait (footfall timing matched to named three-legged gaits),
    peak landing force (x body weight), and how evenly the legs carry load,
    turning and sideways drift, speed fluctuation within a stride, vertical
    bounce, roll and pitch sway and the mean roll (lean).

Results are plotted as heatmaps with speed on the x axis and frequency on the
y axis, and saved as .npz next to the figure.

Usage:
    uv run --frozen scripts/grid_eval.py
    uv run --frozen scripts/grid_eval.py --checkpoint 300 --out logs/sweep_results/it300.png
    uv run --frozen scripts/grid_eval.py --task go2osc_front_locked_omega
"""

import argparse
import textwrap
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
    eval_grid,
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
        "--device",
        type=str,
        default="cpu",
        help="cpu (MuJoCo) or cuda:0 (MuJoCo Warp, as in training; much faster)",
    )
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
        "'irregular'.",
    )
    parser.add_argument(
        "--max_spread",
        type=float,
        default=0.1,
        help="max stride-to-stride circular std of any leg's touchdown "
        "(fraction of a stride); more is 'irregular' whatever the mean timing. "
        "Consistent gaits stay below ~0.05, erratic stepping is above ~0.11.",
    )
    parser.add_argument(
        "--min_stance",
        type=float,
        default=0.04,
        help="contacts shorter than this are taps, ignored when timing the "
        "gait [s]. Taps last 0.01-0.02 s, stances 0.1 s or more.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="figure path; results are saved next to it as .npz (default: "
        "logs/sweep_results/grid_eval.png, or <task>.png with --task)",
    )
    parser.add_argument(
        "--data_out",
        type=str,
        default=None,
        help="where to save the results .npz (default: next to --out)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        name = args.task or "grid_eval"
        args.out = str(Path(GYM_ROOT_DIR) / "logs" / "sweep_results" / f"{name}.png")
    return args


# Rear-first hop: the rear legs land together this fraction of a stride
# before the front legs. Common in the three-legged policies (lead 0.15-0.3
# stride); plain hop is nearer below a lead of about half of it.
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


def robot_label(leg_lock):
    if leg_lock.leg is None:
        return "Intact"
    side = {
        "FL": "Front-left",
        "FR": "Front-right",
        "RL": "Rear-left",
        "RR": "Rear-right",
    }
    return f"{side[leg_lock.leg]} locked ({leg_lock.pose})"


def gait_reference(legs, names):
    """Reference leg, compared legs and named gaits for a robot's active legs.

    Three legs: the first active leg is the reference (named_gaits). Four legs:
    Zhang et al.'s convention, FR reference compared with (FL, RR, RL), i.e.
    their (LF, RH, LH), and their four symmetric gaits.
    """
    if len(legs) == 3:
        return legs[0], list(legs[1:]), named_gaits(names)
    index = dict(zip(names, legs))
    others = [index["FL"], index["RR"], index["RL"]]
    return (
        index["FR"],
        others,
        [
            ("trot", (0.5, 0.5, 0.0)),
            ("pace", (0.5, 0.0, 0.5)),
            ("bound", (0.0, 0.5, 0.5)),
            ("pronk", (0.0, 0.0, 0.0)),
            # Trot whose second diagonal (FL, RR) lands early and split, RR
            # first, while FR and RL stay together (seen in intact policies).
            ("trot, split diagonal", (0.4, 0.25, 0.05)),
        ],
    )


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


def heading(env):
    """Each robot's heading (yaw) [rad]; base_quat is [x, y, z, w]."""
    x, y, z, w = env.base_quat.cpu().numpy().T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_pitch(env):
    """Each robot's roll and pitch [rad]; base_quat is [x, y, z, w]. Roll is
    positive with the right side down, pitch positive nose down."""
    x, y, z, w = env.base_quat.cpu().numpy().T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


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
    peak = torch.zeros(env.num_envs, len(env.feet_indices), device=env.device)
    impulse = torch.zeros_like(peak)
    post_physics_step = env._post_physics_step

    # Peak and impulse of vertical foot force over every physics substep.
    def hooked_post_physics_step():
        post_physics_step()
        force = env.contact_forces[:, env.feet_indices, 2].clamp(min=0.0)
        torch.maximum(peak, force, out=peak)
        impulse.add_(force * sim_dt)

    env._post_physics_step = hooked_post_physics_step
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    contact, step_peak, vx, contact_all = [], [], [], []
    vz, roll, pitch = [], [], []
    n_settle = int(args.settle / env.dt)
    n_steps = int((args.settle + args.measure) / env.dt)
    fell_at = np.full(env.num_envs, n_steps)
    # Turning and sideways drift over the measured window: the turn is summed
    # step by step (so circling past 180 degrees still counts), the sideways
    # offset is measured across the heading held at the window's start.
    turn = np.zeros(env.num_envs)
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
            fell_at[env.terminated.cpu().numpy() & (fell_at == n_steps)] = i
            load = env.contact_forces[:, env.feet_indices, 2] / weight
            contact_all.append((load > args.contact_threshold).cpu().numpy().copy())
            if i == n_settle:
                start_xy = env.root_states[:, :2].cpu().numpy().copy()
                start_yaw = last_yaw = heading(env)
            elif i > n_settle:
                yaw = heading(env)
                turn += (yaw - last_yaw + np.pi) % (2 * np.pi) - np.pi
                last_yaw = yaw
            if i >= n_settle:
                contact.append(contact_all[-1])
                step_peak.append((peak / weight).cpu().numpy().copy())
                vx.append(env.base_lin_vel[:, 0].cpu().numpy().copy())
                vz.append(env.base_lin_vel[:, 2].cpu().numpy().copy())
                r, p = roll_pitch(env)
                roll.append(r)
                pitch.append(p)
    finally:
        env._post_physics_step = post_physics_step
    moved = env.root_states[:, :2].cpu().numpy() - start_xy
    sideways = -np.sin(start_yaw) * moved[:, 0] + np.cos(start_yaw) * moved[:, 1]
    return {
        "turn_rate": np.degrees(np.abs(turn)) / args.measure,  # [robot], deg/s
        "sideways": np.abs(sideways),  # [robot], m over the measured window
        "fell": fell.cpu().numpy(),
        "fell_at": fell_at,  # [robot], first step fallen (n_steps if never)
        "contact_all": np.stack(contact_all, axis=1),  # [robot, step, foot], whole run
        "contact": np.stack(contact, axis=1),  # [robot, step, foot]
        "peak": np.stack(step_peak, axis=1),  # [robot, step, foot], x body weight
        "impulse": impulse.cpu().numpy().copy(),  # [robot, foot], measured window
        "vx": np.stack(vx, axis=1).mean(axis=1),  # [robot]
        "vx_steps": np.stack(vx, axis=1),  # [robot, step], m/s, body frame
        "vz_steps": np.stack(vz, axis=1),  # [robot, step], m/s, body frame
        "roll": np.stack(roll, axis=1),  # [robot, step], rad
        "pitch": np.stack(pitch, axis=1),  # [robot, step], rad
        # Oscillator frequency of each robot [Hz], for one-stride averaging.
        "omega": env.osc_omega[:, 0].cpu().numpy().copy(),
    }


def motion_metrics(log, speed, dt):
    """Per robot: how unsteady the forward speed is within a stride, how much
    the body bounces, sways and pitches, and its mean lean.

    speed_fluctuation: RMS of forward speed about its one-stride moving
        average, as a fraction of the commanded speed (0 = constant speed; a
        stop-start gait such as a pronk is high). Slow drift over several
        strides is removed by the moving average.
    bounce: RMS vertical velocity [m/s].
    roll_sway, pitch_sway: SD of roll and pitch [deg] (oscillation about the
        robot's own mean attitude).
    roll_lean: mean roll [deg], positive with the right side (FR, the locked
        leg) down."""
    n = log["vx_steps"].shape[0]
    out = {k: np.full(n, np.nan) for k in METRIC_KEYS_MOTION}
    for robot in range(n):
        vx = log["vx_steps"][robot]
        period = int(round(1.0 / (max(log["omega"][robot], 1e-3) * dt)))
        if 1 < period < len(vx):
            smooth = np.convolve(vx, np.ones(period) / period, mode="valid")
            start = (period - 1) // 2
            resid = vx[start : start + len(smooth)] - smooth
            out["speed_fluctuation"][robot] = np.sqrt(np.mean(resid**2)) / max(
                speed, 1e-3
            )
        out["bounce"][robot] = np.sqrt(np.mean(log["vz_steps"][robot] ** 2))
        out["roll_sway"][robot] = np.degrees(log["roll"][robot].std())
        out["pitch_sway"][robot] = np.degrees(log["pitch"][robot].std())
        out["roll_lean"][robot] = np.degrees(log["roll"][robot].mean())
    return out


METRIC_KEYS_MOTION = [
    "speed_fluctuation",
    "bounce",
    "roll_sway",
    "pitch_sway",
    "roll_lean",
]


def circular_mean(fractions):
    """Mean over robots of stride fractions, wrapped (0.95 and 0.05 average to 0)."""
    angles = np.exp(2j * np.pi * fractions)
    valid = ~np.isnan(fractions)
    mean = np.where(valid, angles, 0).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    return np.where(valid.any(axis=0), np.angle(mean) / (2 * np.pi) % 1.0, np.nan)


def robot_gait(contact, ref, others, legs, gaits, dt, args, detail=None):
    """One robot's gait from its contact [step, foot], ignoring taps: mean
    touchdown timing of `others` after `ref` (stride fractions), its largest
    stride-to-stride spread, the most touchdowns of a leg per reference
    stride, and the label. Erratic stepping (spread above max_spread) is
    'irregular' even if its mean timing is near a named gait. A three-leg
    robot that barely uses one leg gets a two-leg label (two_leg_label)
    first. With `detail` (a dict), also stores each leg's touchdown count and
    every stride's timing there."""
    min_air = int(round(0.03 / dt))
    contact = footfall.drop_taps(contact, int(round(args.min_stance / dt)))
    starts, fractions = footfall.relative_touchdowns(
        contact, ref, others, min_air, int(round(2.0 / dt))
    )
    lift_fractions = footfall.relative_liftoffs(
        contact, ref, others, min_air, int(round(2.0 / dt))
    )
    timing, spread = circular_timing(fractions, others)
    lift_timing, lift_spread = circular_timing(lift_fractions, others)
    counts = [len(footfall.touchdowns(contact[:, leg], min_air)) for leg in legs]
    rate = [c / max(len(starts), 1) for c in counts]
    lead, pair = np.nan, None
    label = two_leg_label(contact, legs, counts, min_air, dt)
    once = max(counts) > 0 and min(counts) >= ONCE_PER_STRIDE * max(counts)
    if label is None and len(legs) == 3 and max(counts) >= 3:
        pair = steady_pair(contact, legs, counts, min_air, dt, args)
        ratio = min(counts) / max(counts)
        if UNUSED_LEG <= ratio < SUPPORT_LEG:  # third leg used now and then
            steady = pair is not None and pair["stride_cv"] <= SUPPORT_MAX_STRIDE_CV
            label = "2-leg + sporadic 3rd" if steady else "irregular"
        elif not once:
            # A leg steps more often than the others (e.g. twice per stride):
            # no named gait fits; periodic or irregular below.
            label = None
        elif max(spread) <= args.max_spread:
            label, lead = hop_or_bound(timing, ref, others, legs)
    if label is None and (once or len(legs) != 3):
        # Remaining templates; for three legs the hop and the template with
        # the complete pair together are handled by hop_or_bound.
        templates = (
            gaits
            if len(legs) != 3
            else [g for g in gaits if not pair_together_template(g[1], ref, others)]
        )
        label = classify(timing, templates, args.gait_tolerance)
        if label != "not stepping" and max(spread) > args.max_spread:
            label = "irregular"
    if label is None:  # three legs, one leg stepping more often than another
        label = "irregular"
    if label == "irregular" and pair is not None and pair["support"]:
        if pair["stride_cv"] <= SUPPORT_MAX_STRIDE_CV:
            label = "2-leg hop + 3rd-leg support"
    if (
        label == "irregular"
        and max(spread) <= args.max_spread
        and min(counts) >= SUPPORT_LEG * max(counts)
        and stride_cv(starts, dt) <= SUPPORT_MAX_STRIDE_CV
    ):
        # Steady timing, every leg used, steady stride, but no named gait.
        label = PERIODIC
    if (
        len(legs) == 3
        and once
        and label not in ("hop", "irregular", "not stepping", *TWO_LEG_GAITS)
        and max(spread) <= args.max_spread
        and synchronous_liftoff(lift_timing, lift_spread, args.max_spread)
    ):
        # Legs push off together but land one after another (a bound or
        # any other steady timing): a hop with a staggered landing.
        label = STAGGERED
    if detail is not None:
        detail["liftoff_timing"] = np.array(lift_timing)
        detail["liftoff_spread"] = max(lift_spread) if lift_spread else np.nan
        # Share of the measured time each leg is on the ground (duty factor).
        detail["duty"] = contact[:, legs].mean(axis=0)
        detail["support_stride_cv"] = (
            pair["stride_cv"] if pair is not None and pair["support"] else np.nan
        )
        detail["stride_cv"] = pair["stride_cv"] if pair is not None else np.nan
        detail["lead"] = lead
        detail["third_use"] = min(counts) / max(max(counts), 1)
        detail["leg_touchdowns"] = np.array(counts)
        detail["stride_timing"] = np.stack([fractions[leg] for leg in others], axis=-1)
    return np.array(timing), max(spread), max(rate), label


def circular_timing(fractions, legs):
    """Circular mean (in [0, 1)) and std of each leg's per-stride timing,
    as stride fractions; NaN mean with fewer than 3 strides."""
    timing, spread = [], []
    for leg in legs:
        f = fractions[leg][~np.isnan(fractions[leg])]
        z = np.exp(2j * np.pi * f).mean() if len(f) else 0.0
        timing.append(np.angle(z) / (2 * np.pi) % 1.0 if len(f) >= 3 else np.nan)
        spread.append(np.sqrt(-2 * np.log(max(abs(z), 1e-9))) / (2 * np.pi))
    return timing, spread


def synchronous_liftoff(lift_timing, lift_spread, max_spread):
    """True if every leg lifts off within LIFTOFF_SYNC (stride fraction) of
    the reference leg and of each other, consistently stride to stride."""
    t = np.array([0.0, *lift_timing])
    if np.isnan(t).any() or max(lift_spread) > max_spread:
        return False
    gaps = np.abs((t[:, None] - t[None, :] + 0.5) % 1.0 - 0.5)
    return gaps.max() <= LIFTOFF_SYNC


# A leg that lands less than UNUSED_LEG times as often as the busiest leg is
# held up (a two-leg gait). A supporting third leg must land at least
# SUPPORT_LEG times as often; in between it is used only now and then, which
# is left as irregular.
UNUSED_LEG, SUPPORT_LEG = 0.15, 0.4
# The complete pair of a three-leg robot (both legs at one end) lands
# together if at most PAIR_MAX_SPLIT apart. Then the gait is a 'hop' if the
# rear legs land between HOP_MAX_REAR_LEAD before and HOP_MAX_REAR_LAG after
# the front leg(s), else a 'bound'. Both limits sit at dips in the study's
# distribution of rear-leg lead.
PAIR_MAX_SPLIT = 0.2
HOP_MAX_REAR_LEAD, HOP_MAX_REAR_LAG = 0.2, 0.125
# '2-leg hop + 3rd-leg support' also needs a steady stride: the coefficient of
# variation of the pair's stride duration at most this.
SUPPORT_MAX_STRIDE_CV = 0.12  # middle of the gap in a 40-cell sample
# Strides of per-stride timing kept per robot (10 s at 3.75 Hz is ~40).
MAX_STRIDES = 96
PERIODIC = "periodic, unclassified"
# Named gaits need every leg to land about once per stride: the least-used
# leg at least ONCE_PER_STRIDE times as often as the most-used one.
ONCE_PER_STRIDE = 0.75
# Legs that lift off together (all within LIFTOFF_SYNC of a stride) but do not
# land as a hop (e.g. one after another over a quarter stride): a hop whose
# landing is staggered, not a gait with separate beats.
LIFTOFF_SYNC = 0.1
STAGGERED = "hop, staggered landing"
TWO_LEG_GAITS = [
    "2-leg, together",
    "2-leg, alternating",
    "2-leg + sporadic 3rd",
    "2-leg hop + 3rd-leg support",
]


def two_leg_label(contact, legs, counts, min_air, dt):
    """Three-leg robots that run on two legs with the third held up:
    '2-leg, together' or '2-leg, alternating' (by the two legs' mean timing).
    None otherwise, including robots that barely step at all."""
    if len(legs) != 3 or max(counts) < 3:
        return None
    ratio = np.array(counts) / max(counts)
    low = np.flatnonzero(ratio < UNUSED_LEG)
    if len(low) != 1:
        return None
    a, b = [leg for k, leg in enumerate(legs) if k != low[0]]
    _, fractions = footfall.relative_touchdowns(
        contact, a, [b], min_air, int(round(2.0 / dt))
    )
    f = fractions[b][~np.isnan(fractions[b])]
    if len(f) < 3:
        return "2-leg, together"
    lag = np.angle(np.exp(2j * np.pi * f).mean()) / (2 * np.pi)  # in [-0.5, 0.5]
    return "2-leg, together" if abs(lag) < 0.25 else "2-leg, alternating"


def cell_metrics(log, speed, legs, names, dt, args):
    """Per-robot gait, landing peak and load shares, summarized for the cell."""
    ref, others, gaits = gait_reference(legs, names)
    min_air = int(round(0.03 / dt))
    window = max(1, int(round(args.landing_window / dt)))
    ok = ~log["fell"]
    labels, timings, peaks = [], [], []
    robot_spread = np.full(len(ok), np.nan)
    robot_rate = np.full(len(ok), np.nan)
    # Per robot, so gaits can be reclassified later without simulating.
    leg_touchdowns = np.full((len(ok), len(legs)), np.nan)
    robot_lead = np.full(len(ok), np.nan)
    robot_third_use = np.full(len(ok), np.nan)
    robot_stride_cv = np.full(len(ok), np.nan)
    robot_liftoff_timing = np.full((len(ok), len(others)), np.nan)
    robot_liftoff_spread = np.full(len(ok), np.nan)
    robot_duty = np.full((len(ok), len(legs)), np.nan)
    robot_labels = np.full(len(ok), "", dtype=object)
    stride_timing = np.full((len(ok), MAX_STRIDES, len(others)), np.nan)
    for robot in np.flatnonzero(ok):
        contact = log["contact"][robot]
        detail = {}
        timing, robot_spread[robot], robot_rate[robot], label = robot_gait(
            contact, ref, others, legs, gaits, dt, args, detail
        )
        leg_touchdowns[robot] = detail["leg_touchdowns"]
        robot_lead[robot] = detail["lead"]
        robot_third_use[robot] = detail["third_use"]
        robot_stride_cv[robot] = detail["stride_cv"]
        robot_liftoff_timing[robot] = detail["liftoff_timing"]
        robot_liftoff_spread[robot] = detail["liftoff_spread"]
        robot_duty[robot] = detail["duty"]
        robot_labels[robot] = label
        strides = detail["stride_timing"][:MAX_STRIDES]
        stride_timing[robot, : len(strides)] = strides
        timings.append(timing)
        labels.append(label)
        landings = [
            log["peak"][robot, td : td + window, leg].max()
            for leg in legs
            for td in footfall.touchdowns(contact[:, leg], min_air)
        ]
        peaks.append(np.percentile(landings, 95) if landings else np.nan)

    # Every robot's timing (NaN if it fell), so gaits can be reclassified later.
    robot_timing = np.full((len(ok), len(others)), np.nan)
    if timings:
        robot_timing[ok] = np.array(timings)
    shares = log["impulse"][ok][:, legs]
    shares = shares / shares.sum(axis=1, keepdims=True)
    robot_leg_shares = np.full((len(ok), len(legs)), np.nan)
    robot_leg_shares[ok] = shares
    names_list = [name for name, _ in gaits]
    names_list += ["bound", STAGGERED] + TWO_LEG_GAITS if len(legs) == 3 else []
    names_list += [PERIODIC, "irregular", "not stepping"]
    counts = np.array([labels.count(name) for name in names_list])
    robot_motion = motion_metrics(log, speed, dt)
    motion = {
        k: np.median(v[ok]) if ok.any() else np.nan for k, v in robot_motion.items()
    }
    return {
        "speed_ratio": log["vx"][ok].mean() / speed if ok.any() else np.nan,
        "fall_rate": log["fell"].mean(),
        "gait_counts": counts,
        "timing": circular_mean(np.array(timings))
        if timings
        else np.full(len(others), np.nan),
        "robot_timing": robot_timing,
        "robot_spread": robot_spread,
        "robot_touchdowns": robot_rate,
        "robot_leg_touchdowns": leg_touchdowns,
        # Label of every robot (index into the gait names; -1 if it fell).
        "robot_label": np.array(
            [names_list.index(lab) if lab else -1 for lab in robot_labels]
        ),
        "robot_lead": robot_lead,
        "robot_third_use": robot_third_use,
        "robot_stride_cv": robot_stride_cv,
        # Liftoff timing of the other legs after the reference leg's liftoff
        # (stride fractions), its largest stride-to-stride spread, and each
        # leg's share of time on the ground (duty factor).
        "robot_liftoff_timing": robot_liftoff_timing,
        "robot_liftoff_spread": robot_liftoff_spread,
        "robot_duty": robot_duty,
        "robot_leg_shares": robot_leg_shares,
        "robot_stride_timing": stride_timing.astype(np.float32),
        "peak_force": np.nanmean(peaks) if peaks else np.nan,
        "leg_shares": shares.mean(axis=0) if ok.any() else np.full(len(legs), np.nan),
        "unevenness": (shares.std(axis=1) / shares.mean(axis=1)).mean()
        if ok.any()
        else np.nan,
        # Medians over robots that stayed up: how far off a straight line.
        "turn_rate": np.median(log["turn_rate"][ok]) if ok.any() else np.nan,
        "sideways": np.median(log["sideways"][ok]) if ok.any() else np.nan,
        "robot_turn_rate": np.where(ok, log["turn_rate"], np.nan),
        **motion,
        **{f"robot_{k}": np.where(ok, v, np.nan) for k, v in robot_motion.items()},
    }, names_list


def gait_shares_over_time(log, legs, names, gait_names, dt, args):
    """Share of robots in each gait every `args.interval` s over the whole run,
    as in Zhang et al. Fig. 5: RPD averaged over each robot's last two strides,
    nearest named gait within `gait_tolerance`, else irregular."""
    from gait_over_time import classify, last_strides_rpd  # avoids an import cycle

    ref, others, gaits = gait_reference(legs, names)
    gaits = [(n, tuple(2 * np.pi * np.asarray(r))) for n, r in gaits]
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
                    contact[robot, :end], ref, others, min_air, max_stride, 2
                )
                label = classify(rpd, gaits, 2 * np.pi * args.gait_tolerance)
                label = "irregular" if label == "transition" else label
            shares[t, labels.index(label)] += 1
    return times, shares / contact.shape[0], labels


# Reference categorical palette (fixed order) for the named gaits; greys for
# irregular, not stepping and fell, which are not gaits.
GAIT_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#7d5bd0",
    "#9c6b3a",
]
NON_GAIT_COLORS = {
    "periodic, unclassified": "#c9a227",  # ochre: periodic, unlike the greys
    "irregular": "#c8c7c0",
    "not stepping": "#8a8980",
    "fell": "#3d3d3a",
}
# Two-leg gaits: one red family, so they read as related and stand apart.
TWO_LEG_COLORS = {
    # Red, darker the less the third leg is used (gait-mix maps shade by it).
    "2-leg, together": "#a3172f",
    "2-leg + sporadic 3rd": "#dc4b63",
    "2-leg hop + 3rd-leg support": "#f4a3b0",
    "2-leg, alternating": "#6d2d55",  # plum: held up too, but alternating
}
# Hop family in blue (in gait-mix maps the hop is shaded by its rear-leg
# lead; 'rear-first hop' only appears in results from before the hop and the
# rear-first hop were merged), the bound in purple.
HOP_COLORS = {
    "hop": "#1f5fae",
    "rear-first hop": "#8cc0ef",
    "bound": "#7d5bd0",
    "hop, staggered landing": "#4fa3e0",
}
# Kept out of the palette for other gaits when the hop family is shown.
BLUE_PURPLE = {"#2a78d6", "#7d5bd0"}


def observed_legend(names, counts):
    """Legend handles and labels for the gaits with any robots in `counts`
    ([..., gait]); colours come from the full list, so they do not shift."""
    colors = gait_colors(names)
    seen = np.asarray(counts).reshape(-1, len(names)).sum(axis=0) > 0
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, s in zip(colors, seen) if s]
    return handles, [n for n, s in zip(names, seen) if s]


def steady_pair(contact, legs, counts, min_air, dt, args):
    """Two legs landing together consistently (within a quarter stride,
    spread at most max_spread), found by timing rather than landing counts (a
    supporting leg can land more often than the pair, with short extra
    steps). Returns the pair's stride variation (CV of its stride duration)
    and whether the third leg is a 'support' leg: landing at least
    SUPPORT_LEG times as often as the busiest leg but not in steady step with
    the pair (which would be a three-leg gait). None if no pair qualifies."""
    best = None
    for k in range(3):
        a, b = [leg for j, leg in enumerate(legs) if j != k]
        lag, spread = relative_timing(contact, a, b, min_air, dt)
        if abs(lag) < 0.25 and spread <= args.max_spread:
            if best is None or spread < best[0]:
                best = (spread, a, k)
    if best is None:
        return None
    _, a, k = best
    gaps = np.diff(footfall.touchdowns(contact[:, a], min_air)) * dt
    gaps = gaps[gaps <= 2.0]
    cv = gaps.std() / gaps.mean() if len(gaps) > 2 else np.nan
    _, third_spread = relative_timing(contact, a, legs[k], min_air, dt)
    support = counts[k] >= SUPPORT_LEG * max(counts) and not (
        third_spread <= args.max_spread
    )
    return {"stride_cv": cv, "support": support}


def stride_cv(starts, dt):
    """Coefficient of variation of the reference leg's stride duration."""
    gaps = np.diff(starts) * dt
    return gaps.std() / gaps.mean() if len(gaps) > 2 else np.nan


def relative_timing(contact, a, b, min_air, dt):
    """Circular mean (in [-0.5, 0.5)) and spread of when leg b lands in leg
    a's strides; NaN mean and infinite spread with fewer than 3 landings."""
    _, fractions = footfall.relative_touchdowns(
        contact, a, [b], min_air, int(round(2.0 / dt))
    )
    f = fractions[b][~np.isnan(fractions[b])]
    if len(f) < 3:
        return np.nan, np.inf
    z = np.exp(2j * np.pi * f).mean()
    lag = (np.angle(z) / (2 * np.pi) + 0.5) % 1.0 - 0.5
    return lag, np.sqrt(-2 * np.log(max(abs(z), 1e-9))) / (2 * np.pi)


def end_pair(legs):
    """(pair, single) of a three-leg robot: the two legs at the same end
    (front: FL, FR; rear: RL, RR) and the other leg; None if not three legs."""
    if len(legs) != 3:
        return None
    front = [leg for leg in legs if leg < 2]
    rear = [leg for leg in legs if leg >= 2]
    return (rear, front[0]) if len(rear) == 2 else (front, rear[0])


def phases(timing, ref, others):
    """{leg: touchdown time in the reference stride}, reference leg at 0."""
    return {ref: 0.0, **dict(zip(others, np.asarray(timing, float)))}


def hop_or_bound(timing, ref, others, legs):
    """('hop' or 'bound', rear-leg lead) if the complete pair lands
    together (at most PAIR_MAX_SPLIT apart); (None, NaN) otherwise. The lead
    is how far the rear legs land before the front leg(s), in strides
    (negative: after)."""
    pair, single = end_pair(legs)
    t = phases(timing, ref, others)
    split = abs((t[pair[0]] - t[pair[1]] + 0.5) % 1.0 - 0.5)
    if np.isnan(split) or split > PAIR_MAX_SPLIT:
        return None, np.nan
    z = np.exp(2j * np.pi * np.array([t[pair[0]], t[pair[1]]])).mean()
    pair_time = np.angle(z) / (2 * np.pi)
    front, rear = (t[single], pair_time) if pair[0] >= 2 else (pair_time, t[single])
    lead = (front - rear + 0.5) % 1.0 - 0.5
    if -HOP_MAX_REAR_LAG < lead < HOP_MAX_REAR_LEAD:
        return "hop", lead
    return "bound", lead


def pair_together_template(ideal, ref, others):
    """Whether a three-leg template has the complete pair landing together
    (the hop, and e.g. 'RL+RR | FL'): those are classified by hop_or_bound."""
    pair, _ = end_pair(sorted([ref, *others]))
    t = phases(ideal, ref, others)
    return abs(t[pair[0]] - t[pair[1]]) < 1e-9


def gait_colors(labels):
    """Colour per label: fixed colours for gait families (hop, two-leg) and
    non-gaits, the categorical palette in order for the other named gaits."""
    special = NON_GAIT_COLORS | TWO_LEG_COLORS
    if any(label in HOP_COLORS for label in labels):
        special = special | HOP_COLORS
        palette = [c for c in GAIT_COLORS if c not in BLUE_PURPLE]
    else:
        palette = GAIT_COLORS
    named = [label for label in labels if label not in special]
    colors = dict(zip(named, palette))
    colors.update(special)
    return [colors[label] for label in labels]


def plot_over_time(results, labels, path, title, settle):
    """Small multiples: gait share over time per (speed, frequency) cell."""
    speeds, omegas = SWEEP_SPEEDS, SWEEP_OMEGAS
    fig, axes = plt.subplots(
        len(omegas),
        len(speeds),
        figsize=(2.6 * len(speeds) + 2.2, 1.9 * len(omegas)),
        sharex=True,
        sharey=True,
        squeeze=False,
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
                ax.text(
                    0.5,
                    0.5,
                    "not trained",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#6b6a64",
                )
            else:
                times, shares = cell["times"], cell["time_shares"]
                ax.stackplot(
                    times,
                    shares[:, used].T,
                    colors=[c for c, u in zip(colors, used) if u],
                    edgecolor="white",
                    linewidth=0.6,
                )
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
    axes[0, 0].yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, u in zip(colors, used) if u]
    fig.legend(
        handles,
        [l for l, u in zip(labels, used) if u],
        loc="center right",
        fontsize=8,
        frameon=False,
        title="gait",
        title_fontsize=8,
    )
    fig.suptitle(
        title + f"\ncolumns: commanded speed, rows: oscillator frequency; "
        f"dashed line: end of settling ({settle:g} s)",
        fontsize=9,
    )
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
            ax.text(
                col, row, fmt(value), ha="center", va="center", fontsize=8, color=color
            )


def plot(results, gait_names, leg_names, path, title):
    speeds, omegas = np.array(SWEEP_SPEEDS), np.array(SWEEP_OMEGAS)
    shape = (len(omegas), len(speeds))

    def grid(key, reduce=lambda v: v):
        out = np.full(shape, np.nan)
        for (i, j), cell in results.items():
            out[i, j] = reduce(cell[key])
        return out

    # Panels grow with the grid (5 speeds x 4 frequencies at 21 x 13.5).
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(21 * max(1.0, len(speeds) / 5), 13.5 * max(1.0, len(omegas) / 4)),
    )
    panels = [
        # Ranges fitted to the data (see below) so small differences show.
        ("speed_ratio", "measured / commanded speed", "RdBu", None, "{:.2f}"),
        ("fall_rate", "fall rate", "Reds", None, "{:.0%}"),
        (None, "dominant gait (share of robots)", None, None, None),
        ("peak_force", "peak landing force [x body weight]", "plasma", None, "{:.2f}"),
        (
            "unevenness",
            "uneven load across legs [CV, 0 = even]",
            "cividis",
            None,
            "{:.2f}",
        ),
        ("leg_shares", "load share per leg", "Greys", (1 / 3, 1.0), None),
        (
            "turn_rate",
            "turn rate [deg/s] (median; 0 = straight)",
            "magma_r",
            None,
            "{:.1f}",
        ),
        (
            "sideways",
            "sideways drift [m] over the measured window",
            "magma_r",
            None,
            "{:.1f}",
        ),
    ]
    # Panel order: speed, falls, gait, turn rate / peak, evenness, shares, sideways.
    panels = [panels[k] for k in (0, 1, 2, 6, 3, 4, 5, 7)]
    panels += [
        (
            "speed_fluctuation",
            "speed fluctuation within a stride\n[RMS / commanded, 0 = steady]",
            "viridis",
            None,
            "{:.2f}",
        ),
        ("roll_sway", "roll sway [deg, SD]", "viridis", None, "{:.1f}"),
        ("pitch_sway", "pitch sway [deg, SD]", "viridis", None, "{:.1f}"),
        (
            "roll_lean",
            "mean roll [deg, + = locked FR side down]",
            "RdBu_r",
            None,
            "{:.1f}",
        ),
    ]
    for ax, (key, label, cmap, limits, fmt) in zip(axes.ravel(), panels):
        if key is None:  # categorical gait map
            dominant = grid(
                "gait_counts", lambda c: np.argmax(c) if c.sum() else np.nan
            )
            share = grid(
                "gait_counts", lambda c: c.max() / c.sum() if c.sum() else np.nan
            )
            colors = gait_colors(gait_names)
            ax.imshow(
                dominant,
                cmap=ListedColormap(colors),
                vmin=-0.5,
                vmax=len(gait_names) - 0.5,
                origin="lower",
                aspect="auto",
            )
            for (row, col), value in np.ndenumerate(dominant):
                if not np.isnan(value):
                    rgb = matplotlib.colors.to_rgb(colors[int(value)])
                    light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.5
                    ax.text(
                        col,
                        row,
                        f"{textwrap.fill(gait_names[int(value)], 14)}\n{share[row, col]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#1a1a19" if light else "white",
                    )
            handles, seen = observed_legend(
                gait_names, [cell["gait_counts"] for cell in results.values()]
            )
            ax.legend(
                handles,
                seen,
                fontsize=6,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                frameon=False,
            )
        elif key == "leg_shares":
            shares = grid("leg_shares", lambda s: np.nanmax(s))
            ax.imshow(
                shares,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                origin="lower",
                aspect="auto",
            )
            for (i, j), cell in results.items():
                s = cell["leg_shares"]
                if not np.isnan(s).any():
                    text = "\n".join(f"{n} {v:.0%}" for n, v in zip(leg_names, s))
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="tab:red",
                    )
        else:
            values = grid(key)
            vmin, vmax = limits or (np.nanmin(values), np.nanmax(values))
            contrast = key in (
                "speed_ratio",
                "fall_rate",
                "turn_rate",
                "sideways",
                *METRIC_KEYS_MOTION,
            )
            if key in ("turn_rate", "sideways"):
                vmin = 0.0
            if key == "speed_ratio":
                # Diverging around perfect tracking (1.0), symmetric so equal
                # over- and undershoot get equal color strength.
                spread = max(np.nanmax(np.abs(values - 1.0)), 0.05)
                vmin, vmax = 1.0 - spread, 1.0 + spread
            elif key == "fall_rate":
                vmin, vmax = 0.0, max(np.nanmax(values), 0.05)
            elif key == "roll_lean":
                spread = max(np.nanmax(np.abs(values)), 0.5)
                vmin, vmax = -spread, spread
            image = ax.imshow(
                values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
            )
            bar = fig.colorbar(image, ax=ax, shrink=0.85)
            if key == "fall_rate":
                bar.ax.yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(1.0)
                )
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
        play_argv += ["--checkpoint", str(args.checkpoint), "--device", args.device]
        if args.load_run is not None:
            play_argv += ["--load_run", args.load_run]
        with torch.no_grad():
            return play.setup(play.get_play_args(play_argv))

    if args.task is not None:
        # The task's own grid (the wide-range study goes to 4.5 m/s, 6 Hz);
        # rebinds the module-level grid used by the plots below.
        SWEEP_SPEEDS, SWEEP_OMEGAS = eval_grid(args.task)
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
                f"unevenness {cell['unevenness']:.2f}, "
                f"speed fluct {cell['speed_fluctuation']:.2f}, "
                f"roll sway {cell['roll_sway']:.1f} deg"
            )
    if not results:
        raise SystemExit("No trained sweep cells found under logs/")

    path = Path(args.out)
    data_path = Path(args.data_out) if args.data_out else path.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "speed_ratio",
        "fall_rate",
        "peak_force",
        "unevenness",
        "turn_rate",
        "sideways",
        *METRIC_KEYS_MOTION,
    ]
    shape = (len(SWEEP_OMEGAS), len(SWEEP_SPEEDS))
    saved = {key: np.full(shape, np.nan) for key in keys}
    saved["gait_counts"] = np.zeros(shape + (len(gait_names),))
    saved["timing"] = np.full(shape + (len(leg_names) - 1,), np.nan)
    saved["leg_shares"] = np.full(shape + (len(leg_names),), np.nan)
    saved["robot_timing"] = np.full(shape + (args.robots, len(leg_names) - 1), np.nan)
    saved["robot_spread"] = np.full(shape + (args.robots,), np.nan)
    saved["robot_touchdowns"] = np.full(shape + (args.robots,), np.nan)
    saved["robot_turn_rate"] = np.full(shape + (args.robots,), np.nan)
    for key in METRIC_KEYS_MOTION:
        saved[f"robot_{key}"] = np.full(shape + (args.robots,), np.nan)
    saved["robot_leg_touchdowns"] = np.full(
        shape + (args.robots, len(leg_names)), np.nan
    )
    saved["robot_leg_shares"] = np.full(shape + (args.robots, len(leg_names)), np.nan)
    saved["robot_label"] = np.full(shape + (args.robots,), -1, dtype=int)
    for key in (
        "robot_lead",
        "robot_third_use",
        "robot_stride_cv",
        "robot_liftoff_spread",
    ):
        saved[key] = np.full(shape + (args.robots,), np.nan)
    saved["robot_liftoff_timing"] = np.full(
        shape + (args.robots, len(leg_names) - 1), np.nan
    )
    saved["robot_duty"] = np.full(shape + (args.robots, len(leg_names)), np.nan)
    saved["robot_stride_timing"] = np.full(
        shape + (args.robots, MAX_STRIDES, len(leg_names) - 1), np.nan, dtype=np.float32
    )
    times = next(iter(results.values()))["times"]
    saved["time_shares"] = np.full(shape + (len(times), len(time_labels)), np.nan)
    runs = np.full(shape, "", dtype=object)
    for (i, j), cell in results.items():
        for key in saved:
            saved[key][i, j] = cell[key]
        runs[i, j] = cell["run"]
    np.savez(
        data_path,
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
        f"{robot_label(env.cfg.leg_lock)}, "
        + (
            f"{args.task}: one policy for all cells ({runs[0, 0]})"
            if args.task
            else f"one policy per cell ({checkpoint})"
        )
        + f"; {args.robots} robots "
        f"per cell from random phases, measured {args.measure:g} s after "
        f"{args.settle:g} s; physics {sim_hz:.0f} Hz"
    )
    plot(results, gait_names, leg_names, path, title)
    plot_over_time(
        results,
        time_labels,
        path.with_name(path.stem + "_over_time.png"),
        title,
        args.settle,
    )
