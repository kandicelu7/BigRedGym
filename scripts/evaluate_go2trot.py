"""Fixed-command Go2Trot performance gate, independent of training reward.

Run with --help for checkpoint selection. JSON includes every command/frequency
case; --check exits nonzero if any case misses the declared acceptance bounds.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import class_to_dict, get_load_path, set_seed
from gym.utils.task_registry import task_registry


COMMANDS = {
    "stopped": (0.0, 0.0, 0.0),
    "forward_slow": (0.5, 0.0, 0.0),
    "forward": (1.0, 0.0, 0.0),
    "forward_fast": (2.0, 0.0, 0.0),
    "reverse": (-0.5, 0.0, 0.0),
    "left": (0.0, 0.5, 0.0),
    "right": (0.0, -0.5, 0.0),
    "turn_left": (0.0, 0.0, 1.0),
    "turn_right": (0.0, 0.0, -1.0),
    "combined": (1.0, 0.3, 0.5),
}


def high_frequency_rms(value, valid, dt, cutoff_hz=10.0):
    """RMS above the gait harmonics, with detrending and a Hann window.

    Evaluate each surviving trajectory separately; joining replicas or reset
    segments would invent discontinuities. Parseval/window normalization keeps
    the result in the input's physical units.
    """
    powers = []
    for replica in range(value.shape[1]):
        series = value[:, replica][valid[:, replica]].double()
        if len(series) < 4:
            continue
        ramp = torch.linspace(0, 1, len(series))[:, None]
        series = series - (series[0] + ramp * (series[-1] - series[0]))
        window = torch.hann_window(len(series), periodic=False, dtype=torch.double)
        spectrum = torch.fft.rfft(series * window[:, None], dim=0, norm="forward")
        power = spectrum.abs().square() * 2
        power[0] *= 0.5
        if len(series) % 2 == 0:
            power[-1] *= 0.5
        frequency = torch.fft.rfftfreq(len(series), dt)
        powers.append(power[frequency > cutoff_hz].sum(dim=0) / window.square().mean())
    return torch.cat(powers).mean().sqrt().item() if powers else 1e30


def summarize_case(trace, dt, warmup_steps, frequency, command):
    """Summarize [time, replica, feature] traces without rewarding early falls.

    Forces are integrated over complete oscillator strides, separately for
    each replica. Failures are absorbing; their survival gate always fails.
    Derivatives exclude startup and any transition after a failure.
    """
    alive = trace["alive"].bool()
    valid = alive.clone()
    valid[:warmup_steps] = False
    count = int(valid.sum())
    result = {"survival": alive[-1].float().mean().item(), "samples": count}
    if count == 0 or not (valid[1:] & valid[:-1]).any():
        return result | {"passed": False, "failures": ["insufficient valid samples"]}

    def rms(value, mask=valid):
        return value[mask].square().mean().sqrt().item()

    # Planar vector error: averaging over axes would let a stationary policy
    # pass a small nonzero forward command by diluting it with zero lateral error.
    result["linear_rmse_m_s"] = rms(
        (trace["velocity"][..., :2] - command[:2]).norm(dim=-1)
    )
    result["yaw_rmse_rad_s"] = rms(trace["velocity"][..., 2] - command[2])
    pair_valid = valid[1:] & valid[:-1]
    result["residual_step_rms_rad"] = rms(trace["residual"].diff(dim=0), pair_valid)
    result["joint_acceleration_rms_rad_s2"] = rms(
        trace["joint_velocity"].diff(dim=0) / dt, pair_valid
    )
    result["residual_jitter_rms_rad"] = high_frequency_rms(trace["residual"], valid, dt)
    result["joint_velocity_jitter_rms_rad_s"] = high_frequency_rms(
        trace["joint_velocity"], valid, dt
    )
    result["clipped_action_fraction"] = trace["clipped"][valid].float().mean().item()
    result["clipped_action_fraction_by_joint"] = (
        trace["clipped"][valid].float().mean(dim=0).tolist()
    )
    # Count clipping for diagnosis, but gate its physical size: a sub-milliradian
    # overshoot should not count like a policy asking for impossible full turns.
    discarded_rms = trace["clip_error"][valid].square().mean(dim=0).sqrt()
    result["max_discarded_target_rms_rad"] = discarded_rms.max().item()
    result["residual_step_rms_rad_by_joint"] = (
        trace["residual"].diff(dim=0)[pair_valid].square().mean(dim=0).sqrt().tolist()
    )
    loads = trace["load"].clamp_min(0)
    result["foot_support_weight_ratio"] = (
        (loads.sum(dim=-1)[valid] / trace["body_weight"][valid]).mean().item()
    )
    phase = trace["phase"]
    leg_phase = phase + torch.tensor([0.0, torch.pi, torch.pi, 0.0])
    swing = torch.sin(leg_phase) < 0
    total_load = loads[valid].sum().item()
    result["swing_load_fraction"] = (
        (loads * swing)[valid].sum().item() / total_load if total_load else 1.0
    )
    shares = loads[valid].sum(dim=0)
    result["load_share"] = (shares / shares.sum().clamp_min(1e-12)).tolist()
    result["contact_duty"] = (loads[valid] > 5.0).float().mean(dim=0).tolist()

    # Use the known clock rather than wrapped phase differences, including
    # the initial phase offset. Only wholly observed, surviving strides count.
    clock = phase[0, :, 0].double()[None] / (2 * torch.pi)
    clock = clock + torch.arange(len(alive))[:, None] * (dt * frequency)
    cycles = clock.floor().long()
    stride_errors = []
    for replica in range(alive.shape[1]):
        for cycle in cycles[:, replica].unique():
            mask = cycles[:, replica] == cycle
            indices = mask.nonzero().flatten()
            if (
                indices[0] <= warmup_steps
                or indices[-1] == len(alive) - 1
                or not valid[mask, replica].all()
            ):
                continue
            impulse = loads[mask, replica].sum(dim=0) * dt
            share = impulse / impulse.sum().clamp_min(1e-12)
            stride_errors.append((share - 0.25).abs().max())
    result["complete_strides"] = len(stride_errors)
    result["stride_load_max_deviation_p90"] = (
        torch.stack(stride_errors).quantile(0.9).item() if stride_errors else 1.0
    )
    bounds = {
        "survival": (0.95, 1.0),
        "linear_rmse_m_s": (0, 0.20 + 0.1 * command[:2].norm().item()),
        "yaw_rmse_rad_s": (0, 0.35),
        "residual_jitter_rms_rad": (0, 0.02),
        "joint_velocity_jitter_rms_rad_s": (0, 1.0),
        "max_discarded_target_rms_rad": (0, 0.05),
        "foot_support_weight_ratio": (0.8, 1.2),
        "swing_load_fraction": (0, 0.20),
        "stride_load_max_deviation_p90": (0, 0.12),
    }
    failures = [k for k, (lo, hi) in bounds.items() if not lo <= result[k] <= hi]
    if not all(0.25 <= duty <= 0.75 for duty in result["contact_duty"]):
        failures.append("contact_duty")
    if not stride_errors:
        failures.append("complete_strides")
    return result | {"bounds": bounds, "passed": not failures, "failures": failures}


@torch.no_grad()
def evaluate(env, runner, *, replicas=4, seconds=8.0, warmup=2.0):
    cases = [(name, hz) for name in COMMANDS for hz in (1.0, 2.0, 3.0)]
    commands = torch.tensor(
        [COMMANDS[name] for name, _ in cases], device=env.device
    ).repeat_interleave(replicas, dim=0)
    frequencies = torch.tensor(
        [hz for _, hz in cases], device=env.device
    ).repeat_interleave(replicas)
    env.phase_frequency[:, 0] = frequencies
    env.commands[:] = commands
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    records = {
        k: []
        for k in (
            "alive",
            "velocity",
            "residual",
            "joint_velocity",
            "clipped",
            "clip_error",
            "load",
            "phase",
            "body_weight",
        )
    }
    body_weight = env._backend.link_mass.sum(dim=1) * 9.81
    for _ in range(round(seconds / env.dt)):
        # Pin commands even when the task's periodic sampler fires.
        env.commands[:] = commands
        actions = runner.get_inference_actions()
        runner.set_actions(runner.actor_cfg["actions"], actions)
        requested = env.dof_pos_target.clone()
        phase = env.phase.clone()
        env.step()
        alive &= ~env.terminated
        values = {
            "alive": alive,
            "velocity": torch.cat(
                (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1
            ),
            "residual": env.dof_pos_target,
            "joint_velocity": env.dof_vel,
            "clipped": (requested - env.dof_pos_target).abs() > 1e-5,
            "clip_error": requested - env.dof_pos_target,
            "load": env.contact_forces[:, env.feet_indices, 2],
            "phase": phase,
            "body_weight": body_weight,
        }
        for key, value in values.items():
            records[key].append(value.clone())
    trace = {key: torch.stack(value).cpu() for key, value in records.items()}
    if not all(torch.isfinite(value).all() for value in trace.values()):
        raise RuntimeError("Nonfinite evaluation trajectory")
    results = {}
    for i, (name, hz) in enumerate(cases):
        selection = slice(i * replicas, (i + 1) * replicas)
        results[f"{name}@{hz:g}Hz"] = summarize_case(
            {key: value[:, selection] for key, value in trace.items()},
            env.dt,
            round(warmup / env.dt),
            hz,
            torch.tensor(COMMANDS[name]),
        )
        results[f"{name}@{hz:g}Hz"].update(
            command={
                "vx_m_s": COMMANDS[name][0],
                "vy_m_s": COMMANDS[name][1],
                "yaw_rad_s": COMMANDS[name][2],
            },
            gait_frequency_hz=hz,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--warmup", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.replicas < 1 or args.warmup < 0 or args.seconds - args.warmup < 3:
        parser.error("use positive replicas and at least 3 measured seconds")
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            "go2trot",
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    env_cfg.env.num_envs = len(COMMANDS) * 3 * args.replicas
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.push_robots.toggle = False
    apply_domain_randomization_override(env_cfg, "off")
    env_cfg.seed = args.seed
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)
    path = Path(
        get_load_path(
            name=args.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
        )
    )
    env = task_registry.make_env("go2trot", env_cfg, args.device, True, args.backend)
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)
        runner.load(path, load_optimizer=False)
        runner.switch_to_eval()
        results = evaluate(
            env,
            runner,
            replicas=args.replicas,
            seconds=args.seconds,
            warmup=args.warmup,
        )
    finally:
        env._backend.close()
    report = {
        "protocol": "go2trot-v1",
        "jitter_cutoff_hz": 10.0,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree": subprocess.check_output(["git", "status", "--short"], text=True),
        "arguments": vars(args) | {"output": str(args.output)},
        "env_config": class_to_dict(env_cfg),
        "runner_config": class_to_dict(train_cfg),
        "foot_order": list(env.robot_layout.body_groups["feet"]),
        "joint_order": list(env.actuated_dof_names),
        "cases": results,
        "passed": all(case["passed"] for case in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(args.output),
                "passed_cases": sum(c["passed"] for c in results.values()),
                "total_cases": len(results),
            }
        )
    )
    if args.check and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
