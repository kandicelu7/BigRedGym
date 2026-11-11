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
        # if |v_x| <= stop_threshold, else [coupling, offset]. Matches the
        # 0.2 m/s below which the base class zeroes sampled commands.
        stop_threshold = 0.2
        coupling_stop = 4.0
        offset_stop = 1.0
        coupling = 1.0
        offset = 0.0

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

    class commands(Go2Cfg.commands):
        class ranges(Go2Cfg.commands.ranges):
            # Forward speeds only; gaits are evaluated in a forward-speed sweep.
            lin_vel_x = [0.0, 3.0]  # min max [m/s]
            lin_vel_y = 0.0  # max [m/s]
            yaw_vel = 0.0  # max [rad/s]

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 10

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
}
STUDY_SEEDS = range(5)


def study_task_name(setting):
    """Task (and experiment) name of a reward-study setting."""
    return f"go2osc_study_{setting}"


def study_cfgs(setting):
    """Env and runner config classes for one reward-study setting."""
    weights = {**STUDY_BASE_WEIGHTS, **STUDY_SETTINGS[setting]}

    class RunnerCfg(Go2OscFrontLockedOmegaRunnerCfg):
        class critic(Go2OscFrontLockedOmegaRunnerCfg.critic):
            class reward(Go2OscFrontLockedOmegaRunnerCfg.critic.reward):
                class weights(Go2OscFrontLockedOmegaRunnerCfg.critic.reward.weights):
                    pass

        class runner(Go2OscFrontLockedOmegaRunnerCfg.runner):
            experiment_name = study_task_name(setting)

    for name, value in weights.items():
        setattr(RunnerCfg.critic.reward.weights, name, value)
    return Go2OscFrontLockedOmegaCfg, RunnerCfg


STUDY_TASKS = {study_task_name(s): study_cfgs(s) for s in STUDY_SETTINGS}
