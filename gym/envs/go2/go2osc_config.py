from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg


class Go2OscCfg(Go2TrotCfg):
    class env(Go2TrotCfg.env):
        episode_length_s = 10

    class control(Go2TrotCfg.control):
        # Intrinsic oscillator frequency range [Hz], sampled per episode and
        # shared by the four legs. Go2Trot's gait_phase_offsets are unused:
        # each leg has its own oscillator with a random initial phase.
        gait_freq = [1.0, 4.0]

    class osc:
        # Leg to disable (step radius 0), or None for a four-legged robot.
        # One of "FL_foot", "FR_foot", "RL_foot", "RR_foot".
        disabled_leg = "FR_foot"
        # Local load feedback gain [rad/s per unit of normalized foot load]:
        #   d(phase)/dt = radius * (2*pi*f + coupling_sigma * load * cos(phase))
        # A loaded leg is pulled toward mid-stance. 0 gives free-running clocks.
        coupling_sigma = 6.0
        # Weight on ground load carried by a disabled leg in the step_grf reward:
        #   load * (radius * sin(phase) - disabled_leg_contact_penalty * (1 - radius))
        disabled_leg_contact_penalty = 1.0
        # Static pose of a disabled leg, in units of the gait amplitude past
        # the nominal posture (1.0 = the swing apex of a normal step).
        disabled_leg_lift = 2.0

    class viewer(Go2TrotCfg.viewer):
        # Keep the MuJoCo viewer camera on the robot.
        track_robot = True


class Go2OscRunnerCfg(Go2TrotRunnerCfg):
    class actor(Go2TrotRunnerCfg.actor):
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "step_radius",
        ]

    class critic(Go2TrotRunnerCfg.critic):
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
            "phase_frequency",
            "step_radius",
        ]

        class reward:
            # No term refers to a relationship between legs.
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-6
                min_base_height = 0.5
                action_rate = 0.25
                action_rate2 = 0.025
                step_grf = 1.0
                disabled_leg_still = 0.01

            class termination_weight:
                termination = 0.01

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2osc"
        max_iterations = 800
