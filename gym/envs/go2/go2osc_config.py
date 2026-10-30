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
        # [omega, sigma, xi] = [omega_stop, coupling_stop, offset_stop]
        # if |v_x| <= stop_threshold, else
        # [min(omega_step + |v_x|, omega_max), coupling, offset]  (Eq. 5)
        stop_threshold = 0.5
        omega_stop = 1.0  # [Hz]
        coupling_stop = 4.0
        offset_stop = 1.0
        omega_step = 1.5  # [Hz]
        omega_max = 4.0  # [Hz]
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
