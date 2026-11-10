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


class Go2OscFrontLockedOmegaCfg(Go2OscFrontLockedCfg):
    """Front-locked, with the oscillator frequency drawn per episode.

    One policy covers the whole (speed, frequency) grid of the sweep below, so
    gait differences across the grid come from speed and frequency rather than
    from separate training runs. Evaluate with
    scripts/grid_eval.py --task go2osc_front_locked_omega.
    """

    class osc(Go2OscFrontLockedCfg.osc):
        omega_range = [1.25, 4.0]  # [Hz], margin around SWEEP_OMEGAS

    class commands(Go2OscFrontLockedCfg.commands):
        class ranges(Go2OscFrontLockedCfg.commands.ranges):
            # Margin around SWEEP_SPEEDS; above the 0.2 m/s standing cutoff.
            lin_vel_x = [0.25, 2.75]  # [m/s]


class Go2OscFrontLockedOmegaRunnerCfg(Go2OscFrontLockedRunnerCfg):
    class actor(Go2OscFrontLockedRunnerCfg.actor):
        obs = Go2OscFrontLockedRunnerCfg.actor.obs + ["osc_omega"]

    class critic(Go2OscFrontLockedRunnerCfg.critic):
        obs = Go2OscFrontLockedRunnerCfg.critic.obs + ["osc_omega"]

    class runner(Go2OscFrontLockedRunnerCfg.runner):
        experiment_name = "go2osc_front_locked_omega"
        max_iterations = 1500


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
