"""Export a three-legged oscillator policy as a go2_deploy policy bundle.

Writes policy.pt (TorchScript) and deploy_meta.json in the format of the
course's go2_deploy/policy_bundle.py (fixdeploy branch). The policy takes the
scaled observation vector and returns the 12 raw actions that go2_deploy
scales by dof_pos_target and adds to default_dof_pos. The locked leg's three
outputs are replaced by constants that hold it in its lock pose, since in
training they were ignored and the leg was held by its own PD controller.
Pair it with go2_deploy/custom_implementation/tripedal.py.

Usage:
    uv run --frozen scripts/export_course_bundle.py \
        --task go2osc_wide_even2.5_deploy_track --load_run Nov16_16-13-14_seed2 \
        --out exports/tripedal_pronk
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class LockedLegPolicy(torch.nn.Module):
    """Actor (obs normalization included); locked outputs set to the lock pose."""

    def __init__(self, actor, locked, lock_raw):
        super().__init__()
        self.actor = actor
        self.register_buffer("locked", torch.as_tensor(locked, dtype=torch.long))
        self.register_buffer("lock_raw", torch.as_tensor(lock_raw, dtype=torch.float))

    def forward(self, obs):
        action = self.actor.act_inference(obs)
        return action.index_copy(-1, self.locked, self.lock_raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--frequency",
        type=float,
        default=3.0,
        help="default walking oscillator frequency [Hz] (phase_frequency)",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2.0,
        help="forward command limit [m/s] (remote full stick)",
    )
    args = parser.parse_args()

    import play

    import gym.envs.go2.go2osc_config  # noqa: F401  (registers the study tasks)
    from gym.utils.helpers import GYM_ROOT_DIR

    play_argv = [
        "--task",
        args.task,
        "--load_run",
        args.load_run,
        "--headless",
        "--no-keyboard",
        "--no-record",
        "--num_envs",
        "1",
        "--device",
        "cpu",
        "--checkpoint",
        str(args.checkpoint),
    ]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
    names = list(env.actuated_dof_names)
    expected = [
        f"{leg}_{j}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for j in ("hip", "thigh", "calf")
    ]
    if names != expected:
        raise SystemExit(f"joint order {names} is not go2_deploy's {expected}")
    if runner.actor_cfg["actions"] != ["dof_pos_target"]:
        raise SystemExit(f"unexpected actions {runner.actor_cfg['actions']}")
    locked = env._locked_dofs.tolist()
    lock = env.cfg.leg_lock
    lock_pos = torch.tensor(lock.poses[lock.pose])
    scale = torch.as_tensor(env.scales["dof_pos_target"]).flatten().cpu()
    default = env.default_dof_pos.index_select(1, env.actuated_dof_indices)[0].cpu()
    lock_raw = (lock_pos - default[locked]) / scale[locked]

    obs_names = list(runner.actor_cfg["obs"])
    obs_sizes = {n: int(env.get_state(n).shape[1]) for n in obs_names}

    def plain(x):
        return (
            torch.as_tensor(x).flatten().tolist() if hasattr(x, "__len__") else float(x)
        )

    scaling = {n: plain(env.scales.get(n, 1.0)) for n in obs_names + ["dof_pos_target"]}
    for k, v in scaling.items():  # single-element lists back to numbers
        if isinstance(v, list) and len(v) == 1:
            scaling[k] = v[0]

    control = env.cfg.control
    kps = {float(v) for v in control.stiffness.values()}
    kds = {float(v) for v in control.damping.values()}
    if len(kps) != 1 or len(kds) != 1:
        raise SystemExit("go2_deploy takes one kp and one kd for every joint")

    actor = runner.alg.actor
    actor.eval()
    policy = LockedLegPolicy(actor, locked, lock_raw).to("cpu").eval()
    obs = runner.get_obs(obs_names).cpu()[0]
    with torch.no_grad():
        full = actor.act_inference(obs)
        mine = policy(obs)
    free = [i for i in range(12) if i not in locked]
    assert torch.allclose(full[free], mine[free], atol=1e-6), (
        "export changed free joints"
    )
    target = mine[locked] * scale[locked] + default[locked]
    assert torch.allclose(target, lock_pos, atol=1e-5), (
        "locked leg not at its lock pose"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(policy, obs)
    torch.jit.save(traced, out / "policy.pt")
    sha = hashlib.sha256((out / "policy.pt").read_bytes()).hexdigest()
    cmd = env.cfg.commands.ranges
    osc = env.cfg.osc
    lo, hi = osc.omega_range
    if not lo <= args.frequency <= hi:
        raise SystemExit(
            f"--frequency {args.frequency} outside the trained {lo}-{hi} Hz"
        )
    meta = {
        "task_name": args.task,
        "obs_vector": obs_names,
        "obs_sizes": obs_sizes,
        "scaling": scaling,
        "default_dof_pos": default.tolist(),
        "phase_frequency": args.frequency,
        "kp": kps.pop(),
        "kd": kds.pop(),
        "command_limits": {
            "lin_vel_x": min(args.max_speed, float(max(cmd.lin_vel_x))),
            "lin_vel_y": 0.0,
            "yaw_vel": float(cmd.yaw_vel),
        },
        # Oscillator settings for tripedal.py (as in training).
        "osc": {
            "frequency": args.frequency,
            "stand_frequency": float(osc.stand_omega),
            "frequency_rate": float(osc.omega_rate),
            "stop_threshold": float(osc.stop_threshold),
            "coupling": float(osc.coupling),
            "offset": float(osc.offset),
            "coupling_stop": float(osc.coupling_stop),
            "offset_stop": float(osc.offset_stop),
            "substeps": int(env.cfg.control.decimation),
        },
        "num_obs": int(obs.shape[0]),
        "num_actions": 12,
        "policy_sha256": sha,
        "provenance": {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": os.path.relpath(load_path, GYM_ROOT_DIR),
            "custom_implementation": "go2_deploy/custom_implementation/tripedal.py",
            "locked_leg": {
                "joints": [names[i] for i in locked],
                "hold_pos": lock_pos.tolist(),
                "note": "held by the policy's constant outputs at the "
                "bundle kp/kd (training: kp 40, kd 1)",
            },
            "notes": "run with exp_moving_avg = False (no action smoothing in training)",
        },
    }
    (out / "deploy_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Exported {out}: policy.pt, deploy_meta.json ({meta['num_obs']} obs, "
        f"kp {meta['kp']}, kd {meta['kd']}, locked {[names[i] for i in locked]})"
    )


if __name__ == "__main__":
    main()
