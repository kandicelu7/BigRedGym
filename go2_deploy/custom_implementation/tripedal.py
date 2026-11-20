"""Custom implementation for go2_deploy: three-legged Go2 (front-right leg locked)
with decentralized phase oscillators (Zhang et al., ICRA 2024).

Policies (one bundle each, made by scripts/export_course_bundle.py):
- exports/tripedal_pronk/: even2.5_deploy_track_clip_ft2 seed 1 (even-load
  rewards, fine-tuned under 95% motor limits). Pronk at 0.5-1 m/s, irregular
  three-leg stepping at 1.5-2 m/s (3 Hz); bound at 5.25-6 Hz.
- exports/tripedal_trot/: deploy_robust_clip_ft2 seed 1 (baseline rewards,
  fine-tuned under the same limits). Pronk at 0.5-1 m/s, diagonal two-leg gait
  with a light third leg at 1.5-2 m/s (3 Hz).
Everything below mirrors the training env (gym/envs/go2/go2osc.py); oscillator
settings come from the bundle (deploy_meta.json "osc") with the defaults below.

What differs from go2trot:
- The bundle's policy holds the front-right leg in a tucked pose itself (its
  three outputs are constants), so no change to the action path is needed.
- Observation: 8 oscillator phase values and the oscillator frequency, from
  one oscillator per leg driven by the measured foot forces (Tegotae feedback).
  The oscillators are stepped `substeps` times per control step, as in
  training (500 Hz physics: 5 per 10 ms step).
- Commands: forward commands below 0.5 m/s mean stop (stand); the forward
  command ramps at 2 m/s^2; no sideways command; yaw only while moving.
  Remote R1 / L1 raise / lower a held forward speed by 0.5 m/s (0 to the
  bundle's limit), which stays until changed. Pushing the left stick takes
  over and clears the held speed, so releasing the stick stops the robot.
  Each press is printed.
- Oscillator frequency while walking: the bundle's phase_frequency (3 Hz by
  default). Remote Y / B raise / lower it by 0.75 Hz (1.5-6 Hz). A change
  applies the next time the robot is standing (commanded speed below 0.5
  m/s), since the policies were trained with one walking frequency per
  episode; each press is printed.
- Body weight in foot-sensor units is calibrated while the robot stands on
  four legs before the policy starts (INTERMEDIATE, or the first 0.2 s of
  CUSTOM_CTRL if switched straight from RECOVERY).
- Please run with DeployConfig.exp_moving_avg = False (training had no action
  smoothing); kp, kd and the default pose come from the bundle.
"""

import math

import torch

# --- Defaults (the bundle's "osc" entry overrides them) -----------------------
OSC = {
    "frequency": 3.0,  # [Hz] walking frequency (bundle: phase_frequency)
    "stand_frequency": 1.0,  # [Hz] while stopped (holds the loaded legs' phases)
    "frequency_rate": 4.0,  # [Hz/s] frequency ramp
    "stop_threshold": 0.5,  # [m/s] forward commands below this mean stop
    "coupling": 1.0,  # Tegotae feedback while moving
    "offset": 0.0,
    "coupling_stop": 4.0,  # ... while stopped
    "offset_stop": 1.0,
    "substeps": 5,  # oscillator steps per control step (physics / control rate)
}
FREQUENCY_STEP = 0.75  # [Hz] per Y / B press
FREQUENCY_RANGE = (1.5, 6.0)  # [Hz] (trained 1.25-6.25)
MAX_ACCEL = 2.0  # [m/s^2] forward-command ramp
MAX_YAW = 1.0  # [rad/s]
SPEED_STEP = 0.5  # [m/s] per R1 / L1 press (held forward speed)
STICK_DEADZONE = 0.1  # [m/s] left-stick speed below this leaves the held speed
DT = 0.01  # control period [s] (100 Hz, as DeployConfig.ctrl_freq)
CALIBRATION_STEPS = 20  # control steps of four-leg standing averaged (0.2 s)

# Oscillators and observations use QGym's foot order FL, FR, RL, RR; the
# front-right (index 1) is locked and its oscillator is masked out.
ACTIVE = torch.tensor([1.0, 0.0, 1.0, 1.0])
# lowstate_msg.foot_force is in Unitree order FR, FL, RR, RL.
UNITREE_FOOT_TO_QGYM = [1, 0, 3, 2]
# QGym joint indices of the locked leg (FR hip, thigh, calf).
LOCKED_DOFS = [3, 4, 5]


class CustomDeployConfig:
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "phase_obs",
        "osc_omega",
    ]

    # Length of observation vector pieces (the others are standard)
    custom_obs_sizes = {
        "phase_obs": 8,  # cos(phase) x 4 legs, then sin(phase) x 4 legs
        "osc_omega": 1,  # oscillator frequency [Hz]
    }

    # Observation = value / scale (as in training). The bundle's
    # deploy_meta.json carries the same values, and the bundle's win.
    class DeployScaling:
        base_ang_vel = 0.3
        projected_gravity = 1.0
        commands = [3.0, 1.0, 3.0]
        dof_pos_obs = [1.0472, 2.53075, 0.94247] * 4
        dof_vel = [2.0, 2.0, 4.0] * 4
        dof_pos_target = [0.5236, 1.265375, 0.471235] * 4
        phase_obs = 1.0
        osc_omega = 4.0


# Initialize any custom buffers your implementation uses (for 1 environment)
def get_custom_buffers_dict():
    return {
        "osc_params": None,  # OSC merged with the bundle's, on the first update
        "osc_phase": torch.rand(4) * 2 * math.pi,  # [rad] per leg (FL, FR, RL, RR)
        "osc_omega": OSC["stand_frequency"],  # current frequency [Hz]
        "osc_walk_frequency": OSC["frequency"],  # in use while walking [Hz]
        "osc_next_frequency": OSC["frequency"],  # set by Y / B, applied when standing
        "osc_fresh": True,  # first custom-control step: frequency jumps to target
        "osc_command": torch.zeros(3),  # command after stop threshold and ramp
        "osc_weight": None,  # foot-sensor reading equal to the body weight
        "osc_weight_samples": [],
        "osc_buttons": (0, 0),  # last (Y, B) state, for press detection
        "osc_warned": False,
        "held_speed": 0.0,  # [m/s] set by R1 / L1
        "speed_buttons": (0, 0),  # last (R1, L1) state
    }


def _params(mc):
    """Oscillator settings: defaults, then the bundle's (deploy_meta.json)."""
    if mc.osc_params is None:
        p = dict(OSC)
        meta = getattr(mc.cfg, "bundle_meta", None) or {}
        p.update(meta.get("osc", {}))
        if getattr(mc.cfg, "phase_frequency", None):
            p["frequency"] = float(mc.cfg.phase_frequency)
        mc.osc_params = p
        mc.osc_walk_frequency = mc.osc_next_frequency = p["frequency"]
        print(
            f"[tripedal] walking frequency {p['frequency']:g} Hz "
            f"(Y/B: +/-{FREQUENCY_STEP:g} Hz, applied when standing; "
            f"R1/L1: held speed +/-{SPEED_STEP:g} m/s); "
            f"oscillator substeps {p['substeps']}",
            flush=True,
        )
    return mc.osc_params


def _calibrate(mc, msg):
    """Average the four foot forces while the robot stands on four legs."""
    total = float(sum(msg.foot_force))
    if total <= 0:
        return
    mc.osc_weight_samples.append(total)
    if len(mc.osc_weight_samples) >= CALIBRATION_STEPS:
        mc.osc_weight = sum(mc.osc_weight_samples) / len(mc.osc_weight_samples)
        print(
            f"[tripedal] body weight = {mc.osc_weight:.1f} foot-sensor units",
            flush=True,
        )


def _buttons(mc, p, standing):
    """Y / B presses change the walking frequency (applied when standing)."""
    rc = mc.remote_controller
    now = (int(rc.Y[0]), int(rc.B[0]))
    step = (now[0] and not mc.osc_buttons[0]) - (now[1] and not mc.osc_buttons[1])
    mc.osc_buttons = now
    if step:
        lo, hi = FREQUENCY_RANGE
        mc.osc_next_frequency = min(
            hi, max(lo, mc.osc_next_frequency + step * FREQUENCY_STEP)
        )
        when = "now" if standing else "at the next stop"
        print(
            f"[tripedal] walking frequency -> {mc.osc_next_frequency:g} Hz ({when})",
            flush=True,
        )
    if standing:
        mc.osc_walk_frequency = mc.osc_next_frequency


def _speed_buttons(mc, max_speed):
    """R1 / L1 presses raise / lower the held forward speed."""
    rc = mc.remote_controller
    now = (int(rc.R1[0]), int(rc.L1[0]))
    step = (now[0] and not mc.speed_buttons[0]) - (now[1] and not mc.speed_buttons[1])
    mc.speed_buttons = now
    if step:
        mc.held_speed = min(max_speed, max(0.0, mc.held_speed + step * SPEED_STEP))
        note = "  (stand)" if mc.held_speed < OSC["stop_threshold"] else ""
        print(f"[tripedal] held speed -> {mc.held_speed:g} m/s{note}", flush=True)


# Define how your custom buffers should be updated at the control frequency
def update_custom_buffers(main_controller, t):
    mc = main_controller
    msg = mc.last_lowstate_msg
    if msg is None:
        return
    p = _params(mc)
    if not mc.osc_warned:
        mc.osc_warned = True
        if getattr(mc.cfg, "exp_moving_avg", False):
            print(
                "[tripedal] WARNING: exp_moving_avg is on; these policies were "
                "trained without action smoothing (robots fell in simulation "
                "with it on). Set DeployConfig.exp_moving_avg = False.",
                flush=True,
            )
    if mc._state.name != "CUSTOM_CTRL":
        # Standing on four legs before the policy starts: calibrate, and start
        # the next run from rest.
        if mc.osc_weight is None:
            _calibrate(mc, msg)
        mc.osc_command = torch.zeros(3)
        mc.osc_fresh = True
        mc.held_speed = 0.0  # never start walking on a speed from an earlier run
        return
    if mc.osc_weight is None:  # switched straight from RECOVERY: still standing
        _calibrate(mc, msg)

    # Commands (Go2Osc._resample_commands / _ramp_commands): forward only,
    # below the threshold means stop, and no turning while stopped.
    rc = mc.remote_controller
    max_speed = float(mc.cfg.command_limits.get("lin_vel_x", 2.0))
    _speed_buttons(mc, max_speed)
    stick = rc.lin_vel_x
    if abs(stick) > STICK_DEADZONE:
        # The stick takes over and clears the held speed, so releasing it
        # stops the robot.
        vx = stick
        if mc.held_speed:
            mc.held_speed = 0.0
            print("[tripedal] held speed -> 0 m/s (stick)", flush=True)
    else:
        vx = mc.held_speed
    vx = min(max(vx, 0.0), max_speed)
    stop_target = vx < p["stop_threshold"]
    target = 0.0 if stop_target else vx
    cmd = mc.osc_command.clone()
    dv = MAX_ACCEL * DT
    cmd[0] += max(-dv, min(dv, target - cmd[0].item()))
    cmd[1] = 0.0
    cmd[2] = 0.0 if stop_target else max(-MAX_YAW, min(MAX_YAW, rc.yaw_vel))
    mc.osc_command = cmd

    # Oscillator parameters (Go2Osc._update_osc_params).
    stop = abs(cmd[0].item()) < p["stop_threshold"]
    _buttons(mc, p, stop)
    goal = p["stand_frequency"] if stop else mc.osc_walk_frequency
    if mc.osc_fresh:
        mc.osc_omega = goal
        mc.osc_fresh = False
    else:
        df = p["frequency_rate"] * DT
        mc.osc_omega += max(-df, min(df, goal - mc.osc_omega))
    coupling = p["coupling_stop"] if stop else p["coupling"]
    offset = p["offset_stop"] if stop else p["offset"]

    # Oscillator steps (Go2Osc._step_oscillators, without the training noise),
    # `substeps` per control step with the latest foot loads.
    load = _foot_load(mc, msg)
    n = int(p["substeps"])
    h = DT / n
    phase = mc.osc_phase
    for _ in range(n):
        feedback = coupling * load * (torch.cos(phase) + offset)
        phase = torch.remainder(
            phase + 2 * math.pi * (mc.osc_omega - feedback) * h, 2 * math.pi
        )
    mc.osc_phase = phase


def _foot_load(mc, msg):
    """Foot force / body weight in QGym foot order, clamped to [0, 1] as in
    training (Go2Osc._foot_load); zero until the weight is calibrated."""
    if not mc.osc_weight:
        return torch.zeros(4)
    force = torch.tensor([float(f) for f in msg.foot_force])[UNITREE_FOOT_TO_QGYM]
    return (force / mc.osc_weight).clamp(0.0, 1.0)


# Return a dict mapping observation names to functions that generate the observation pieces
def get_custom_obs_pieces():
    return {
        "commands": _get_obs_commands,
        "dof_pos_target": _get_obs_dof_pos_target,
        "phase_obs": _get_obs_phase_obs,
        "osc_omega": _get_obs_osc_omega,
    }


# Commands after the stop threshold and ramp (replaces the raw remote command)
def _get_obs_commands(main_controller, lowstate_msg):
    return main_controller.osc_command.clone()


# Previous action, with the locked leg's entries zeroed as in training
def _get_obs_dof_pos_target(main_controller, lowstate_msg):
    target = main_controller.last_action[0].clone()
    target[LOCKED_DOFS] = 0.0
    return target


# cos and sin of each leg's phase; the locked leg's entries are 0
def _get_obs_phase_obs(main_controller, lowstate_msg):
    phase = main_controller.osc_phase
    return torch.cat((torch.cos(phase), torch.sin(phase))) * ACTIVE.repeat(2)


# Oscillator frequency [Hz]
def _get_obs_osc_omega(main_controller, lowstate_msg):
    return torch.tensor([float(main_controller.osc_omega)])
