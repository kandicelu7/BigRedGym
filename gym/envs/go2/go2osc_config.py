from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg

# Side used by the locked-leg experiments: "R" locks FR / RR, "L" locks FL / RL.
LOCKED_SIDE = "R"


class Go2OscCfg(Go2Cfg):
    """Go2 with decentralized phase oscillators (Zhang et al. 2024).

    One oscillator per leg, coupled only to its own foot load:
        dphi_i/dt = 2 pi (omega - sigma * F_i * (cos(phi_i) + xi))
    Swing is phi in [0, pi), stance is phi in [pi, 2 pi).
    """

    class env(Go2Cfg.env):
        # Long enough for the oscillators to settle into a gait (~10 s).
        episode_length_s = 10

    class osc:
        process_noise_std = 0.25
        # Frequency rises continuously with commanded speed (no jump at the
        # stop threshold, unlike Eq. 5), so slow walking gets slow stepping:
        #   omega = min(omega_min + omega_slope * |v_x|, omega_max)
        omega_min = 1.0  # [Hz] at |v_x| = 0
        # 1.25 gives the 3.5 Hz of Eq. 5 at 2 m/s; 1.5 hit the cap there and
        # the policy overshot fast commands (2 m/s -> ~3 m/s).
        omega_slope = 1.25  # [Hz per m/s]
        omega_max = 4.0  # [Hz]
        # [low, high] in Hz: instead of the schedule above, draw omega
        # uniformly per robot at each reset and hold it for the episode, so one
        # policy learns every (speed, frequency) pair. None uses the schedule.
        omega_range = None
        # Foot load as the oscillators see it: Gaussian noise [x body weight]
        # and a delay drawn per episode from 0..grf_delay_max_steps physics
        # steps. Zero for both is the exact simulated load.
        grf_noise_std = 0.0
        grf_delay_max_steps = 0
        # Standing regime (Eq. 5): [sigma, xi] = [coupling_stop, offset_stop]
        # if |v_x| < stop_threshold, else [coupling, offset]. Matches the
        # 0.2 m/s below which the base class zeroes sampled commands.
        stop_threshold = 0.2
        coupling_stop = 4.0
        offset_stop = 1.0
        coupling = 1.0
        offset = 0.0
        # Start/stop (deploy settings). stand_omega [Hz]: frequency while
        # stopped, low enough (below about coupling_stop x foot load x 2) for
        # the standing coupling to hold the phases still; None keeps the
        # moving frequency. omega_rate [Hz/s]: the frequency ramps toward its
        # target at this rate instead of jumping; None jumps.
        stand_omega = None
        omega_rate = None

    class leg_lock:
        # Leg held in a fixed pose to emulate a failure: None (intact), or one
        # of "FL", "FR", "RL", "RR".
        leg = None
        pose = "tucked"
        # Absolute [hip, thigh, calf] joint angles [rad].
        poses = {
            # Calf fully folded, thigh raised, so the foot clears the ground.
            "tucked": [0.0, 1.36, -2.65],
            # Calf at full extension, roughly under the hip: a passive peg.
            "straight": [0.0, 0.42, -0.85],
        }
        stiffness = 40.0
        damping = 1.0

    class motor_limits:
        # Actuator limits as fractions of the URDF limits (Go2Osc._apply_motor_limits).
        # torque_scale: every joint torque is clipped to this share of its
        # effort limit (the base env clips at 1.0). speed_scale: top speed as
        # a share of the velocity limit; the torque that speeds a joint up
        # tapers to zero there and overspeed is braked. None: no speed limit.
        torque_scale = 1.0
        speed_scale = None
        taper_start = 0.75  # share of the top speed where torque starts to fall
        # [Nm per rad/s] of overspeed. Off: at 100 Hz physics it made joints
        # chatter; with 500 Hz physics the taper alone holds the top speed.
        brake_gain = 0.0

    class commands(Go2Cfg.commands):
        class ranges(Go2Cfg.commands.ranges):
            # Forward speeds only; gaits are evaluated in a forward-speed sweep.
            lin_vel_x = [0.0, 3.0]  # min max [m/s]
            lin_vel_y = 0.0  # max [m/s]
            yaw_vel = 0.0  # max [rad/s]

        # Start/stop curriculum [first, last] control step (8 per training
        # iteration): no stop commands or at-rest starts before `first`, then
        # zero_prob, rest_prob and the share of slow commands that become stop
        # grow linearly to full at `last` (slow commands that do not stop are
        # raised to osc.stop_threshold). None: full from the start.
        startstop_ramp = None

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 10
        # Control steps over which pushes grow to full size (0: no ramp).
        ramp_steps = 0

    class asset(Go2Cfg.asset):
        penalize_contacts_on = ["calf", "hip"]
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]

    class reward_settings(Go2Cfg.reward_settings):
        base_height_target = 0.25
        # Foot load [x body weight] above which a landing is penalized
        # (peak_load reward); 1.0 is the robot's weight on a single foot.
        peak_load_threshold = 1.0
        # Heading error [rad] at which the heading reward falls to 1/e.
        heading_sigma = 0.25
        # Orientation reward: exp(-g_x^2 / pitch_sigma) + exp(-g_y^2 /
        # roll_sigma), g the gravity direction in the body frame. 0.25 for
        # both is the base class's reward (which uses tracking_sigma); a
        # larger roll_sigma tolerates more sideways sway.
        pitch_sigma = 0.25
        roll_sigma = 0.25
        # Steady-speed reward: forward-speed deviation from its one-stride
        # running average, in units of steady_sigma x the commanded speed
        # (at least 0.5 m/s), squared and capped at steady_cap. Off for
        # steady_delay [s] after each reset and new command; the cap bounds
        # what is left of speeding up to a new command, and slow commands.
        steady_sigma = 0.1
        steady_delay = 1.0
        steady_cap = 4.0
        # Power-smoothness reward: relative deviation of joint power from its
        # one-stride average, the average at least power_floor [W], squared
        # and capped; off for power_smooth_delay [s] after a reset.
        power_floor = 50.0
        power_smooth_cap = 4.0
        power_smooth_delay = 1.0
        # Cost-of-transport reward: commanded speed at least this [m/s].
        cot_min_speed = 0.5
        # Flight reward: fewer than flight_min_feet active feet with load above
        # contact_threshold [x body weight] is penalized, fully below
        # flight_speed[0] and fading out at flight_speed[1] [m/s].
        contact_threshold = 0.05
        flight_min_feet = 2
        flight_speed = [1.0, 1.5]
        # Stand-still reward: off for stand_grace [s] after a stop command,
        # then phased in over the next stand_grace seconds.
        stand_grace = 0.5
        # Joint-speed penalty (weight joint_speed_limit): starts at this share
        # of each joint's velocity limit.
        speed_soft_limit = 0.85

    class scaling(Go2Cfg.scaling):
        phase_vel = 2 * 3.14159 * 4.0
        osc_omega = 4.0


class Go2OscRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
        ]

    class critic(Go2RunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_vel",
        ]

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                orientation = 1.0
                min_base_height = 1.0
                torques = 5.0e-6
                action_rate = 0.25
                action_rate2 = 0.025
                collision = 0.5
                gait = 1.0
                # Off by default; set in the reward study's tasks.
                even_load = 0.0
                peak_load = 0.0
                heading = 0.0
                steady_speed = 0.0
                power_smooth = 0.0
                cost_of_transport = 0.0
                flight = 0.0
                stand_still = 0.0
                joint_speed_limit = 0.0

            class termination_weight:
                termination = 0.15

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2osc"
        max_iterations = 1500


class Go2OscFrontLockedCfg(Go2OscCfg):
    class leg_lock(Go2OscCfg.leg_lock):
        leg = "F" + LOCKED_SIDE


class Go2OscFrontLockedRunnerCfg(Go2OscRunnerCfg):
    class runner(Go2OscRunnerCfg.runner):
        experiment_name = "go2osc_front_locked"


class Go2OscHindLockedCfg(Go2OscCfg):
    class leg_lock(Go2OscCfg.leg_lock):
        leg = "R" + LOCKED_SIDE


class Go2OscHindLockedRunnerCfg(Go2OscRunnerCfg):
    class runner(Go2OscRunnerCfg.runner):
        experiment_name = "go2osc_hind_locked"


def omega_cfgs(base_cfg, base_runner_cfg, experiment_name):
    """Variant of a task with the oscillator frequency drawn per episode.

    One policy covers the whole (speed, frequency) grid of the sweep below, so
    gait differences across the grid come from speed and frequency rather than
    from separate training runs. Evaluate with
    scripts/grid_eval.py --task <experiment_name>.
    """

    class Cfg(base_cfg):
        class osc(base_cfg.osc):
            omega_range = [1.25, 4.0]  # [Hz], margin around SWEEP_OMEGAS

        class commands(base_cfg.commands):
            class ranges(base_cfg.commands.ranges):
                # Margin around SWEEP_SPEEDS; above the 0.2 m/s standing cutoff.
                lin_vel_x = [0.25, 2.75]  # [m/s]

    class RunnerCfg(base_runner_cfg):
        class actor(base_runner_cfg.actor):
            obs = base_runner_cfg.actor.obs + ["osc_omega"]

        class critic(base_runner_cfg.critic):
            obs = base_runner_cfg.critic.obs + ["osc_omega"]

        class runner(base_runner_cfg.runner):
            max_iterations = 1500

    RunnerCfg.runner.experiment_name = experiment_name
    return Cfg, RunnerCfg


# The three robots of the study, each over the whole (speed, frequency) grid.
Go2OscOmegaCfg, Go2OscOmegaRunnerCfg = omega_cfgs(
    Go2OscCfg, Go2OscRunnerCfg, "go2osc_omega"
)
Go2OscFrontLockedOmegaCfg, Go2OscFrontLockedOmegaRunnerCfg = omega_cfgs(
    Go2OscFrontLockedCfg, Go2OscFrontLockedRunnerCfg, "go2osc_front_locked_omega"
)
Go2OscHindLockedOmegaCfg, Go2OscHindLockedOmegaRunnerCfg = omega_cfgs(
    Go2OscHindLockedCfg, Go2OscHindLockedRunnerCfg, "go2osc_hind_locked_omega"
)


# Front-locked variants.
class Go2OscFrontLockedOmegaGait2RunnerCfg(Go2OscFrontLockedOmegaRunnerCfg):
    """Gait reward doubled, so following the oscillators is worth more and
    the frequency may have more say over the gait."""

    class critic(Go2OscFrontLockedOmegaRunnerCfg.critic):
        class reward(Go2OscFrontLockedOmegaRunnerCfg.critic.reward):
            class weights(Go2OscFrontLockedOmegaRunnerCfg.critic.reward.weights):
                gait = 2.0

    class runner(Go2OscFrontLockedOmegaRunnerCfg.runner):
        experiment_name = "go2osc_front_locked_omega_gait2"


class Go2OscFrontLockedOmegaStraightCfg(Go2OscFrontLockedOmegaCfg):
    """Locked leg held straight, as a passive peg, instead of tucked."""

    class leg_lock(Go2OscFrontLockedOmegaCfg.leg_lock):
        pose = "straight"


class Go2OscFrontLockedOmegaStraightRunnerCfg(Go2OscFrontLockedOmegaRunnerCfg):
    class runner(Go2OscFrontLockedOmegaRunnerCfg.runner):
        experiment_name = "go2osc_front_locked_omega_straight"


class Go2OscFrontLockedOmegaRobustCfg(Go2OscFrontLockedOmegaCfg):
    """First step toward hardware: the oscillators run on a noisy, delayed
    foot load (as from real foot sensors), with wider friction and mass
    randomization. The gait reward still uses the true load."""

    class osc(Go2OscFrontLockedOmegaCfg.osc):
        grf_noise_std = 0.1  # [x body weight]
        grf_delay_max_steps = 2  # per-episode delay of 0..2 physics steps

    class domain_randomization(Go2OscFrontLockedOmegaCfg.domain_randomization):
        class startup(Go2OscFrontLockedOmegaCfg.domain_randomization.startup):
            contact_friction_range = [0.3, 1.2]
            link_mass_scale_range = [0.85, 1.15]


class Go2OscFrontLockedOmegaRobustRunnerCfg(Go2OscFrontLockedOmegaRunnerCfg):
    class actor(Go2OscFrontLockedOmegaRunnerCfg.actor):
        add_noise = True  # sensor noise on the policy's observations

    class runner(Go2OscFrontLockedOmegaRunnerCfg.runner):
        experiment_name = "go2osc_front_locked_omega_robust"


# Coarse sweep: one front-locked policy per (commanded speed, oscillator
# frequency) cell, each trained at that fixed speed and frequency with
# feedback on, so each cell's gait emerges independently.
SWEEP_SPEEDS = [0.5, 1.0, 1.5, 2.0, 2.5]  # [m/s]
SWEEP_OMEGAS = [1.5, 2.25, 3.0, 3.75]  # [Hz]
SWEEP_ITERATIONS = 400


def sweep_task_name(speed, omega):
    """Task (and experiment) name of a sweep cell, e.g. go2osc_front_locked_v1.5_w3"""
    return f"go2osc_front_locked_v{speed:g}_w{omega:g}"


def sweep_cfgs(speed, omega):
    """Env and runner config classes for one sweep cell."""

    class Cfg(Go2OscFrontLockedCfg):
        class osc(Go2OscFrontLockedCfg.osc):
            # Fixed frequency, independent of the command.
            omega_min = omega
            omega_slope = 0.0
            omega_max = max(omega, Go2OscFrontLockedCfg.osc.omega_max)

        class commands(Go2OscFrontLockedCfg.commands):
            class ranges(Go2OscFrontLockedCfg.commands.ranges):
                lin_vel_x = [speed, speed]

    class RunnerCfg(Go2OscFrontLockedRunnerCfg):
        class runner(Go2OscFrontLockedRunnerCfg.runner):
            experiment_name = sweep_task_name(speed, omega)
            max_iterations = SWEEP_ITERATIONS

    return Cfg, RunnerCfg


SWEEP_TASKS = {
    sweep_task_name(speed, omega): sweep_cfgs(speed, omega)
    for omega in SWEEP_OMEGAS
    for speed in SWEEP_SPEEDS
}


# Reward study: how reward weights shape the gaits a front-locked policy
# learns over the whole (speed, frequency) grid. One factor at a time around
# a shared baseline, five seeds per setting (train.py --seed 0..4).
# Every setting, including the baseline, also rewards holding the initial
# heading: without it the three-legged robots run in curves (30-130 degrees
# of turn in 10 s), which the yaw-rate reward barely notices.
STUDY_BASE_WEIGHTS = {"gait": 1.0, "tracking_lin_vel": 4.0, "heading": 2.0}
STUDY_SETTINGS = {
    "baseline": {},
    # Gait reward x1.5, x2.
    "gait1.5": {"gait": 1.5},
    "gait2": {"gait": 2.0},
    # Velocity tracking and heading together ("go where commanded") x1.5, x2.
    "track1.5": {"tracking_lin_vel": 6.0, "heading": 3.0},
    "track2": {"tracking_lin_vel": 8.0, "heading": 4.0},
    # Even load and low peak load, off (0) in the baseline. Weights from the
    # trained baseline-like policies: each term is then about half of, and
    # about equal to, the gait reward (raw even_load ~ -0.1, peak_load
    # ~ -0.045 per step).
    "even2.5": {"even_load": 2.5},
    "even5": {"even_load": 5.0},
    "peak6": {"peak_load": 6.0},
    "peak12": {"peak_load": 12.0},
}
STUDY_SEEDS = range(5)
# The wide-range study adds settings that change the orientation reward and
# reward a steady speed. A "reward_settings" entry overrides
# Go2OscCfg.reward_settings instead of a weight.
WIDE_SETTINGS = {
    **STUDY_SETTINGS,
    # Orientation: roll tolerated 4x more (sway), or the whole orientation
    # reward halved (both sway and pitch).
    "roll4": {"reward_settings": {"roll_sigma": 1.0}},
    "orient0.5": {"orientation": 0.5},
    # Steady speed, off (0) in the baseline. Raw value for the trained wide
    # baseline under training commands ~ -0.34 per step (forward speed varies
    # by ~3-5% of the command within a stride above 1 m/s, more when slow),
    # so these are about half of, and about equal to, the gait reward (~0.36).
    "steady0.5": {"steady_speed": 0.5},
    "steady1": {"steady_speed": 1.0},
}

# Follow-up settings of the wide study (two training batches). Besides
# weights and "reward_settings", a setting may contain
#   "push": push_robots overrides (stronger, more frequent pushes, grown
#       from zero over ramp_steps control steps; 1500 iterations of 8 steps
#       are 12000);
#   "friction": the contact-friction range (training default [0.5, 1.0]);
#   "sensors": True for noisy, delayed foot load for the oscillators, wider
#       link-mass randomization and observation noise (as the robust task).
# Push test pushes are 0.5-2 m/s; training pushes are at most 0.5 m/s per
# axis every 10 s by default.
ROBUST_PUSH = {"max_push_vel_xy": 1.25, "interval_s": 4.0, "ramp_steps": 4000}
ROBUST_FRICTION = [0.2, 1.25]
ROBUST = {"push": ROBUST_PUSH, "friction": ROBUST_FRICTION, "sensors": True}
# Deployable start/stop behaviour, also usable as setting entries:
#   "osc": osc overrides; "commands": command overrides (zero_prob: share of
#       new commands that are 'stop'; max_accel [m/s^2]: forward-command
#       ramp; yaw_vel [rad/s]: yaw-rate command range); "init_state":
#       rest_prob, the share of resets that start at rest.
# Commands below 0.5 m/s mean stop: the oscillators drop to 1 Hz (slow
# enough for the standing coupling to freeze the loaded legs' phases),
# ramping at 4 Hz/s; a quarter of commands are stop, the forward command
# ramps at 2 m/s^2, a third of episodes start at rest, and the policy turns
# at up to 1 rad/s while moving. stand_still penalizes joint motion from
# 0.5-1 s after a stop command (raw ~ -25 when stepping in place).
DEPLOY = {
    "push": ROBUST_PUSH,
    "friction": ROBUST_FRICTION,
    "stand_still": 0.04,
    "osc": {"stop_threshold": 0.5, "stand_omega": 1.0, "omega_rate": 4.0},
    "commands": {"zero_prob": 0.25, "max_accel": 2.0, "yaw_vel": 1.0},
    "init_state": {"rest_prob": 0.3},
}
# The hardware candidate: DEPLOY plus noisy, delayed foot loads for the
# oscillators, wider mass randomization and observation noise (as robust),
# with forward commands limited to 2.5 m/s.
DEPLOY_ROBUST = {
    **DEPLOY,
    "sensors": True,
    "commands": {**DEPLOY["commands"], "lin_vel_x": [0.25, 2.5]},
}
# Stronger speed and heading tracking (as even2.5_track2: fixed the deploy
# policy's ~10% speed undershoot and cut 2 m/s push falls from 7% to 1%).
DEPLOY_TRACK = {**DEPLOY_ROBUST, "tracking_lin_vel": 8.0, "heading": 4.0}
# even2.5_deploy seed 4, the one deploy policy that always stands still when
# stopped (start/stop test); committed so molab can start from it.
DEPLOY_SEED4 = (
    "checkpoints/go2osc_wide_even2.5_deploy__Nov16_09-10-17_seed4_model_2000.pt"
)
# Actuator limits at 95% of the URDF torque and speed limits (motor_limits):
# the policies above exceed the calf's 15.7 rad/s and hit the hip/thigh
# torque limit (joint-limit check, 2026-11-18).
# 500 Hz physics: at 100 Hz a light joint overshoots its top speed within one
# 10 ms step (calf to ~2x its limit); at 500 Hz the taper holds it (<= 92%).
LIMITS95 = {
    "motor_limits": {"torque_scale": 0.95, "speed_scale": 0.95},
    "sim_hz": 500,
}
# Starting points for fine-tuning under those limits: deploy_robust seed 2
# (closest to the limits) and the hardware candidate, even2.5_deploy_track seed 2.
DEPLOY_ROBUST_SEED2 = (
    "checkpoints/go2osc_wide_deploy_robust__Nov16_23-40-18_seed2_model_2000.pt"
)
DEPLOY_ROBUST_SEED4 = (
    "checkpoints/go2osc_wide_deploy_robust__Nov17_00-14-30_seed4_model_2000.pt"
)
DEPLOY_TRACK_SEED2 = (
    "checkpoints/go2osc_wide_even2.5_deploy_track__Nov16_16-13-14_seed2_model_2000.pt"
)
FOLLOWUP_BATCHES = {
    # Batch 1: sim-to-real candidates and the power-smoothness hypothesis
    # (even load blocks two-leg gaits, smooth power blocks the hop).
    1: {
        "robust": ROBUST,
        "even2.5_robust": {"even_load": 2.5, **ROBUST},
        "even2.5_smooth0.5": {"even_load": 2.5, "power_smooth": 0.5},
        "even2.5_smooth1": {"even_load": 2.5, "power_smooth": 1.0},
    },
    # Batch 2: pushes and friction separately, no flight phase at low speed
    # (walking gaits), and a mild cost of transport on top of even load.
    2: {
        "push": {"push": ROBUST_PUSH},
        "friction": {"friction": ROBUST_FRICTION},
        "even2.5_noflight1": {"even_load": 2.5, "flight": 1.0},
        "even2.5_cot0.2": {"even_load": 2.5, "cost_of_transport": 0.2},
    },
    # Batch 3: a deployable policy (even load, pushes, wide friction, plus
    # starting from rest, stopping below 0.5 m/s and turning while moving),
    # and whether speed tracking on top of even load finds a three-leg gait.
    3: {
        "even2.5_deploy": {"even_load": 2.5, **DEPLOY},
        "even2.5_track2": {"even_load": 2.5, "tracking_lin_vel": 8.0, "heading": 4.0},
    },
    # Batch 4: the hardware candidate (deploy with sensor noise, to 2.5 m/s).
    4: {
        "even2.5_deploy_robust": {"even_load": 2.5, **DEPLOY_ROBUST},
    },
    # Batch 5: fine-tune deploy seed 4 with sensor noise, the 2.5 m/s cap and
    # stronger tracking (init_from: its weights, fresh optimizer, iterations
    # counted from 0); and the same from scratch with a stronger stand-still
    # penalty, since 0.04 left 3 of 5 deploy seeds stepping in place.
    5: {
        "even2.5_deploy_ft4": {
            "even_load": 2.5,
            **DEPLOY_TRACK,
            "init_from": DEPLOY_SEED4,
        },
        "even2.5_deploy_track": {"even_load": 2.5, **DEPLOY_TRACK, "stand_still": 0.1},
    },
    # Batch 6: the sim-to-real package on the baseline rewards (no even load,
    # no tracking boost), to compare with even2.5_deploy_robust; the
    # stand-still penalty as in deploy_track.
    6: {
        "deploy_robust": {**DEPLOY_ROBUST, "stand_still": 0.1},
    },
    # Batch 7: fine-tune the two policies above under hard limits at 95% of
    # the actuator torque and speed limits (train 1000 iterations): do they
    # keep their gaits within the limits?
    7: {
        "deploy_robust_clip_ft2": {
            **DEPLOY_ROBUST,
            "stand_still": 0.1,
            **LIMITS95,
            "init_from": DEPLOY_ROBUST_SEED2,
        },
        "even2.5_deploy_track_clip_ft2": {
            "even_load": 2.5,
            **DEPLOY_TRACK,
            "stand_still": 0.1,
            **LIMITS95,
            "init_from": DEPLOY_TRACK_SEED2,
        },
    },
    # Batch 8: the two hardware policies in one stage: batch 7's settings
    # (95% motor limits, 500 Hz physics) trained from scratch for 2000
    # iterations instead of fine-tuned, with full-size pushes from the start
    # (ramp_steps 0: the base push code, nothing else needed), so the two
    # go2osc files alone reproduce them. tripedal_pronk: even load (the
    # baseline-reward version is HARDWARE_TASKS["go2osc_baseline"]).
    8: {
        "tripedal_pronk": {
            "even_load": 2.5,
            **DEPLOY_TRACK,
            "stand_still": 0.1,
            **LIMITS95,
            "push": {**ROBUST_PUSH, "ramp_steps": 0},
            "max_iterations": 2000,
        },
    },
}
# Hardware tasks under their own names (registered with STUDY_TASKS; the
# evaluation scripts treat them like wide-study tasks). go2osc_baseline: the
# baseline rewards + start/stop and sim-to-real training (DEPLOY_ROBUST,
# stand-still 0.1) + 95% torque/speed limits at 500 Hz physics, trained from
# scratch for 2000 iterations with full-size pushes from the start (the base
# push code), so go2osc.py and this file alone reproduce it.
HARDWARE_TASKS = {
    "go2osc_baseline": {
        **DEPLOY_ROBUST,
        "stand_still": 0.1,
        **LIMITS95,
        "push": {**ROBUST_PUSH, "ramp_steps": 0},
        "max_iterations": 2000,
    },
    # go2osc_baseline with curricula (go2osc_baseline itself, trained from
    # scratch, learned a two-leg hop on FL+RL): pushes grow to full size over
    # the first 500 iterations (ROBUST_PUSH), and stop commands and at-rest
    # starts are off until iteration 800, then grow to full by 1200 (8
    # control steps per iteration).
    "go2osc_baseline_curriculum": {
        **DEPLOY_ROBUST,
        "stand_still": 0.1,
        **LIMITS95,
        "push": ROBUST_PUSH,
        "commands": {**DEPLOY_ROBUST["commands"], "startstop_ramp": [800 * 8, 1200 * 8]},
        "max_iterations": 2000,
    },
}
# The same curricula at 100 Hz physics (about 1.5x faster to train). At 100 Hz
# the speed taper alone lets joints overshoot (a calf passes its limit within
# one physics step), so the policy is also penalized for joint speeds above
# 85% of the limit (joint_speed_limit); torque stays clipped at 95%. Check
# peak joint speeds at 500 Hz after training. 2500 iterations: the earlier
# 100 Hz runs levelled off by ~1000, and start/stop is at full strength only
# from 1200. track: stronger speed and heading tracking (DEPLOY_TRACK);
# even: that plus the even-load reward (as tripedal_pronk).
LIMITS95_100HZ = {
    "motor_limits": LIMITS95["motor_limits"],
    "joint_speed_limit": 10.0,
    "reward_settings": {"speed_soft_limit": 0.85},
}
HARDWARE_TASKS["go2osc_baseline_curriculum_100hz"] = {
    **HARDWARE_TASKS["go2osc_baseline_curriculum"],
    "sim_hz": None,
    **LIMITS95_100HZ,
    "max_iterations": 2500,
}
HARDWARE_TASKS["go2osc_track_curriculum_100hz"] = {
    **HARDWARE_TASKS["go2osc_baseline_curriculum_100hz"],
    "tracking_lin_vel": DEPLOY_TRACK["tracking_lin_vel"],
    "heading": DEPLOY_TRACK["heading"],
}
HARDWARE_TASKS["go2osc_even_curriculum_100hz"] = {
    **HARDWARE_TASKS["go2osc_track_curriculum_100hz"],
    "even_load": 2.5,
}
WIDE_SETTINGS.update({k: v for b in FOLLOWUP_BATCHES.values() for k, v in b.items()})
# Each follow-up setting's reference: the setting it adds one change to.
FOLLOWUP_PARENTS = {
    k: ("even2.5" if k.startswith("even2.5_") else "baseline")
    for b in FOLLOWUP_BATCHES.values()
    for k in b
}


# Wide-range study: the same settings over speeds to 4.5 m/s and oscillator
# frequencies to 6 Hz. 4.5 m/s is about the canine amputees' top speed (6.7
# m/s at 0.61 m withers height) at the same Froude number for the Go2's
# ~0.3 m base height; 6 Hz keeps the stride length at the top speed near
# that of the original grid (2.5 m/s at 3.75 Hz, ~0.7 m). The grid contains
# every cell of the original one, so the overlap can be compared directly.
WIDE_SPEEDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # [m/s]
WIDE_OMEGAS = [1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0]  # [Hz]


class Go2OscFrontLockedWideCfg(Go2OscFrontLockedOmegaCfg):
    class osc(Go2OscFrontLockedOmegaCfg.osc):
        omega_range = [1.25, 6.25]  # [Hz], margin around WIDE_OMEGAS

    class commands(Go2OscFrontLockedOmegaCfg.commands):
        class ranges(Go2OscFrontLockedOmegaCfg.commands.ranges):
            lin_vel_x = [0.25, 4.75]  # [m/s], margin around WIDE_SPEEDS


def study_task_name(setting, wide=False):
    """Task (and experiment) name of a reward-study setting."""
    return f"go2osc_{'wide' if wide else 'study'}_{setting}"


def eval_grid(task):
    """(speeds, frequencies) at which grid_eval.py evaluates a task."""
    if task.startswith("go2osc_wide_") or task in HARDWARE_TASKS:
        return WIDE_SPEEDS, WIDE_OMEGAS
    return SWEEP_SPEEDS, SWEEP_OMEGAS


def study_cfgs(setting, wide=False):
    """Env and runner config classes for one reward-study setting (or a
    HARDWARE_TASKS name: wide, logged under that name)."""
    hardware = setting in HARDWARE_TASKS
    changed = dict(
        HARDWARE_TASKS[setting]
        if hardware
        else (WIDE_SETTINGS if wide else STUDY_SETTINGS)[setting]
    )
    wide = wide or hardware
    reward_settings = changed.pop("reward_settings", {})
    push = changed.pop("push", None)
    friction = changed.pop("friction", None)
    sensors = changed.pop("sensors", False)
    osc = changed.pop("osc", {})
    commands = changed.pop("commands", {})
    init_state = changed.pop("init_state", {})
    init_from = changed.pop("init_from", None)
    motor_limits = changed.pop("motor_limits", {})
    sim_hz = changed.pop("sim_hz", None)  # physics rate [Hz] (policy rate unchanged)
    max_iterations = changed.pop("max_iterations", None)  # default for train.py
    weights = {**STUDY_BASE_WEIGHTS, **changed}

    class RunnerCfg(Go2OscFrontLockedOmegaRunnerCfg):
        class critic(Go2OscFrontLockedOmegaRunnerCfg.critic):
            class reward(Go2OscFrontLockedOmegaRunnerCfg.critic.reward):
                class weights(Go2OscFrontLockedOmegaRunnerCfg.critic.reward.weights):
                    pass

        class runner(Go2OscFrontLockedOmegaRunnerCfg.runner):
            experiment_name = setting if hardware else study_task_name(setting, wide)

    if init_from is not None:
        RunnerCfg.runner.init_from = init_from  # loaded by scripts/train.py
    if max_iterations is not None:
        RunnerCfg.runner.max_iterations = max_iterations
    for name, value in weights.items():
        setattr(RunnerCfg.critic.reward.weights, name, value)
    if sensors:
        RunnerCfg.actor = type(
            "actor", (Go2OscFrontLockedOmegaRunnerCfg.actor,), {"add_noise": True}
        )
    env_cfg = Go2OscFrontLockedWideCfg if wide else Go2OscFrontLockedOmegaCfg
    if (
        reward_settings
        or push
        or friction
        or sensors
        or osc
        or commands
        or init_state
        or motor_limits
        or sim_hz
    ):
        base = env_cfg

        class EnvCfg(base):
            class reward_settings(base.reward_settings):
                pass

            class commands(base.commands):
                class ranges(base.commands.ranges):
                    pass

            class init_state(base.init_state):
                pass

            class push_robots(base.push_robots):
                pass

            class osc(base.osc):
                pass

            class motor_limits(base.motor_limits):
                pass

            class control(base.control):
                pass

            class domain_randomization(base.domain_randomization):
                class startup(base.domain_randomization.startup):
                    pass

        for name, value in reward_settings.items():
            setattr(EnvCfg.reward_settings, name, value)
        for name, value in (push or {}).items():
            setattr(EnvCfg.push_robots, name, value)
        for name, value in osc.items():
            setattr(EnvCfg.osc, name, value)
        for name, value in commands.items():
            ranged = name in ("yaw_vel", "lin_vel_x", "lin_vel_y")
            target = EnvCfg.commands.ranges if ranged else EnvCfg.commands
            setattr(target, name, value)
        for name, value in init_state.items():
            setattr(EnvCfg.init_state, name, value)
        for name, value in motor_limits.items():
            setattr(EnvCfg.motor_limits, name, value)
        startup = EnvCfg.domain_randomization.startup
        if friction is not None:
            startup.contact_friction_range = list(friction)
        if sensors:
            robust = Go2OscFrontLockedOmegaRobustCfg
            EnvCfg.osc.grf_noise_std = robust.osc.grf_noise_std
            EnvCfg.osc.grf_delay_max_steps = robust.osc.grf_delay_max_steps
            startup.link_mass_scale_range = list(
                robust.domain_randomization.startup.link_mass_scale_range
            )
        if sim_hz:
            # Physics-step quantities keep their meaning in seconds: the foot
            # load delay (in steps) and the oscillator noise (added per step).
            ratio = sim_hz / base.control.desired_sim_frequency
            EnvCfg.control.desired_sim_frequency = sim_hz
            EnvCfg.osc.grf_delay_max_steps = int(
                round(EnvCfg.osc.grf_delay_max_steps * ratio)
            )
            EnvCfg.osc.process_noise_std = EnvCfg.osc.process_noise_std * ratio**0.5
        env_cfg = EnvCfg
    return env_cfg, RunnerCfg


STUDY_TASKS = {
    **{study_task_name(s): study_cfgs(s) for s in STUDY_SETTINGS},
    **{study_task_name(s, True): study_cfgs(s, True) for s in WIDE_SETTINGS},
    **{name: study_cfgs(name) for name in HARDWARE_TASKS},
}
