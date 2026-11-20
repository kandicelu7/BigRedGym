# Tripedal Go2: policies for hardware testing

A Go2 with its front-right leg locked (tucked) walks on three legs using
decentralized phase oscillators (Zhang et al., ICRA 2024). Two policies:

| Bundle | Gaits at 3 Hz | Other gaits |
|---|---|---|
| `tripedal_pronk/` | pronk at 0.5-1 m/s, irregular three-leg stepping at 1.5-2 m/s | bound at 5.25-6 Hz |
| `tripedal_trot/` | pronk at 0.5-1 m/s, diagonal two-leg gait (FL+RR, rear-left only taps) at 1.5-2 m/s | pronk at 4.5-6 Hz |

Each bundle (go2_deploy format, `policy_bundle.load_bundle`) holds `policy.pt`
(TorchScript, 54 observations -> 12 actions) and `deploy_meta.json`
(observation layout and scaling, default pose, kp 20 / kd 0.5, command
limits, oscillator settings). Custom implementation:
`go2_deploy/custom_implementation/tripedal.py`.

## Training (task names, iterations, commits)

| Policy | Stage 1 | Stage 2 (fine-tune) |
|---|---|---|
| `tripedal_pronk` | `go2osc_wide_even2.5_deploy_track`, 2000 iterations | `go2osc_wide_even2.5_deploy_track_clip_ft2`, 1000 iterations |
| `tripedal_trot` | `go2osc_wide_deploy_robust`, 2000 iterations | `go2osc_wide_deploy_robust_clip_ft2`, 1000 iterations |

All runs: 4096 robots, GPU (MuJoCo Warp). Stage 1 uses 100 Hz physics; stage 2
uses 500 Hz physics with torque clipped to 95% of each joint's limit and the
torque that speeds a joint up tapering to zero at 95% of its speed limit
(`motor_limits`, `sim_hz` in `go2osc_config.py`).

The bundled policies were trained as follows (the task configs at the PR head
are identical to those at these commits):

| Policy | Stage 1 run | Stage 2 run |
|---|---|---|
| `tripedal_pronk` | seed 2, commit `f96973b` | seed 1, commit `6bc2360` |
| `tripedal_trot` | seed 2, commit `3f93f65` | seed 1, commit `6bc2360` |

Stage 2 starts from the stage 1 runs above, committed in `checkpoints/`, so a
fresh fine-tune is one command:

    uv run --frozen scripts/train.py --task go2osc_wide_even2.5_deploy_track_clip_ft2 \
        --num_envs 4096 --max_iterations 1000 --device cuda:0 --headless

Both stages fresh, e.g. for `tripedal_pronk`:

    uv run --frozen scripts/train.py --task go2osc_wide_even2.5_deploy_track \
        --num_envs 4096 --max_iterations 2000 --device cuda:0 --headless
    uv run --frozen scripts/train.py --task go2osc_wide_even2.5_deploy_track_clip_ft2 \
        --init_from logs/go2osc_wide_even2.5_deploy_track/<run>/model_2000.pt \
        --num_envs 4096 --max_iterations 1000 --device cuda:0 --headless

(`tripedal_trot`: `go2osc_wide_deploy_robust`, then `go2osc_wide_deploy_robust_clip_ft2`.)
`notebooks/molab_batch7.py` runs stage 2 for 5 seeds and evaluates them on molab.

**Seeds matter.** GPU training is not bit-for-bit repeatable, and results vary
by seed: of 5 stage-1 seeds of the even-load setting, one (seed 2) passed every
criterion. Train several seeds and pick with the evaluations below.

Export a bundle from a trained run:

    uv run --frozen scripts/export_course_bundle.py --task <task> --load_run <run> \
        --out exports/<name> [--frequency 3.0] [--max_speed 2.0]

## Evaluation (simulation)

One run (`--task <task> --load_run <run>`):

    uv run --frozen scripts/grid_eval.py --task <task> --load_run <run> --device cuda:0   # speed x frequency grid, gaits
    uv run --frozen scripts/push_eval.py --task <task> --load_run <run> --device cuda:0
    uv run --frozen scripts/startstop_eval.py --task <task> --load_run <run> --device cuda:0

Both bundles were checked closed-loop in simulation through the go2_deploy code
(`load_bundle`, `lowstate_to_obs`, `RLController`) with this custom
implementation: no falls, tracking at 1 and 1.5 m/s, clean stop and restart,
frequency change, R1/L1 held speed and stick takeover. At 0.5-2 m/s both
stay within the joint speed and torque limits (measured at 3, 4.5 and 6 Hz for
`tripedal_pronk`, 3.75 Hz for `tripedal_trot`).

## Customizations (vs. go2trot)

- **Observation (54):** base_ang_vel (3), projected_gravity (3), commands (3),
  dof_pos_obs (12), dof_vel (12), dof_pos_target (12, locked leg's entries 0),
  phase_obs (8: cos then sin of each leg's oscillator phase, front-right 0),
  osc_omega (1: oscillator frequency in Hz). Scaling in `deploy_meta.json`.
- **Oscillators** run in the custom implementation from the measured foot forces
  (Tegotae feedback), 5 steps per 10 ms control step as in training. The body
  weight in foot-sensor units is calibrated while the robot stands on four legs
  (INTERMEDIATE, or the first 0.2 s after A from RECOVERY). Assumed
  `foot_force` order: FR, FL, RR, RL (please check by lifting each foot).
- **Action space:** unchanged (12 joint targets), but the policy's three
  front-right outputs are constants that hold the leg tucked at
  [0, 1.36, -2.65] rad.
- **Commands:** forward only, up to 2.0 m/s; below 0.5 m/s means stop and stand;
  the forward command ramps at 2 m/s^2; no sideways command; turning (up to
  1 rad/s) only while walking.
- **Gains and pose:** kp 20, kd 0.5, thigh default 0.66 rad (from the bundle).
- **Action smoothing must be off:** `DeployConfig.exp_moving_avg = False`
  (training had none; every simulated robot fell with it on).

## Controls

| Control | Action |
|---|---|
| Left stick, forward | Forward speed (full stick = 2.0 m/s). Pushing it clears the held speed, so releasing stops the robot |
| Right stick, sideways | Turn rate (walking only) |
| R1 / L1 | Held forward speed +/- 0.5 m/s (0-2.0), kept until changed; reset to 0 whenever the policy is not running |
| Y / B | Walking frequency +/- 0.75 Hz (1.5-6 Hz, default 3), applied when the robot is standing |
| A, X, arrows, F1, L1+R1 / L2+R2 held | Course defaults (start policy, e-stop, kp/kd, ABXY toggle, launcher start/stop). Please leave kp/kd at the bundle values |

## Suggested test plan

| Policy | Frequency | Speed | Expected gait |
|---|---|---|---|
| `tripedal_pronk` | 3 Hz | 0.5-1 m/s | pronk |
| `tripedal_pronk` | 3 Hz | 1.5-2 m/s | irregular three-leg stepping |
| `tripedal_pronk` | 5.25-6 Hz | 1-2 m/s | bound |
| `tripedal_trot` | 3 Hz | 1.5-2 m/s | two-leg gait with a light third leg |
| `tripedal_trot` | 4.5 Hz | 0.5-1 m/s | pronk |
