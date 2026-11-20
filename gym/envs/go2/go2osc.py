import torch

from gym.envs.go2.go2 import Go2
from gym.utils.math.simple_math import wrap_to_pi
from gym.utils.sampling import masked_update


def yaw_of(quat_xyzw):
    """Heading [rad] of quaternions in the task layer's [x, y, z, w] order."""
    x, y, z, w = quat_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Go2Osc(Go2):
    """Go2 with one decentralized phase oscillator per leg.

    Follows Zhang et al., "Learning Emergent Gaits with Decentralized Phase
    Oscillators" (2024). Optionally holds one leg in a fixed pose
    (cfg.leg_lock) to emulate a leg failure.
    """

    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_leg_lock()

        # Oscillators follow the canonical foot order (FL, FR, RL, RR).
        self.phase = torch.zeros(self.num_envs, 4, device=self.device)
        self.phase_vel = torch.zeros_like(self.phase)
        self.phase_obs = torch.zeros(self.num_envs, 8, device=self.device)
        self.grf = torch.zeros_like(self.phase)
        self.osc_omega = torch.ones(self.num_envs, 1, device=self.device)
        # Per-episode frequency, used instead of the schedule if osc.omega_range.
        self.episode_omega = torch.ones_like(self.osc_omega)
        # Recent foot loads, newest first, for the delayed load the
        # oscillators see (osc.grf_delay_max_steps).
        self._load_history = torch.zeros(
            self.num_envs, self.cfg.osc.grf_delay_max_steps + 1, 4, device=self.device
        )
        self._load_delay = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.osc_coupling = torch.zeros_like(self.osc_omega)
        self.osc_offset = torch.zeros_like(self.osc_omega)

        # State for the even-load, peak-load and heading rewards. Updated
        # every step whether or not those rewards are weighted (reward
        # functions only run when they are).
        # Each foot's load [x body weight] averaged over about one stride.
        self.stride_load = torch.zeros_like(self.phase)
        # Largest load of each foot during the current control step, over all
        # physics substeps (a landing impact lasts only a few substeps).
        self.peak_load = torch.zeros_like(self.phase)
        # Heading the robot should keep [rad]: its heading at reset, turned
        # by any yaw-rate command.
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        # Forward speed averaged over about one stride (exponential moving
        # average with time constant 1 / omega), for the steady-speed reward.
        self.stride_vx = torch.zeros(self.num_envs, device=self.device)
        # Joint power |torque x joint speed| summed over joints [W], averaged
        # over the current control step, and its one-stride running average
        # (power-smoothness and cost-of-transport rewards).
        self.step_power = torch.zeros(self.num_envs, device=self.device)
        self.stride_power = torch.zeros(self.num_envs, device=self.device)
        # Share of the control step's physics substeps with fewer than
        # reward_settings.flight_min_feet active feet on the ground.
        self.step_flight = torch.zeros(self.num_envs, device=self.device)
        # Start/stop: the forward-speed command the ramp moves toward
        # (commands.max_accel), time [s] the command has been 'stop', and
        # robots reset since the last oscillator update.
        self.command_target = torch.zeros(self.num_envs, device=self.device)
        self.stop_time = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_leg_lock(self):
        lock = self.cfg.leg_lock
        foot_names = self.robot_layout.body_groups["feet"]
        if lock.leg is not None and not any(
            name.startswith(lock.leg + "_") for name in foot_names
        ):
            raise ValueError(f"unknown leg_lock.leg {lock.leg!r}")
        # The locked leg's oscillator is meaningless; mask it out.
        self.active_legs = torch.tensor(
            [
                lock.leg is None or not name.startswith(lock.leg + "_")
                for name in foot_names
            ],
            dtype=torch.bool,
            device=self.device,
        )

        locked = [
            i
            for i, name in enumerate(self.actuated_dof_names)
            if lock.leg is not None and name.startswith(lock.leg + "_")
        ]
        self._locked_dofs = torch.tensor(locked, dtype=torch.long, device=self.device)
        self._locked_full_dofs = self.actuated_dof_indices[self._locked_dofs]
        # Locked joints are ordered hip, thigh, calf like the pose entries.
        self._lock_pos = torch.tensor(
            lock.poses[lock.pose] if locked else [],
            dtype=torch.float,
            device=self.device,
        )
        self._lock_torque_limits = self.actuated_torque_limits[self._locked_dofs]

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.peak_load.zero_()
        self.step_power.zero_()
        self.step_flight.zero_()
        # The policy has no authority over the locked leg; zero its targets so
        # observations and action-rate penalties ignore it.
        self.dof_pos_target[:, self._locked_dofs] = 0.0
        self._ramp_commands()
        self._update_osc_params()

    def _update_osc_params(self):
        """Oscillator parameters from the commanded forward speed.

        Frequency grows continuously with speed (or is the episode's
        frequency, with osc.omega_range); coupling and offset switch to the
        standing regime of Eq. 5 below osc.stop_threshold. With
        osc.stand_omega the frequency drops to it while stopped, low enough
        for the standing coupling to hold every loaded leg's phase still; with
        osc.omega_rate [Hz/s] it moves toward its target at that rate instead
        of jumping (except right after a reset).
        """
        osc = self.cfg.osc
        speed = torch.abs(self.commands[:, 0:1])
        if osc.omega_range is not None:
            target = self.episode_omega
        else:
            target = torch.clamp(
                osc.omega_min + osc.omega_slope * speed, max=osc.omega_max
            )
        stop = speed < osc.stop_threshold
        if osc.stand_omega is not None:
            target = torch.where(stop, osc.stand_omega, target)
        if osc.omega_rate is not None:
            step = osc.omega_rate * self.dt
            ramped = self.osc_omega + torch.clamp(target - self.osc_omega, -step, step)
            target = torch.where(self._fresh.unsqueeze(1), target, ramped)
        self.osc_omega[:] = target
        self._fresh[:] = False
        self.osc_coupling[:] = torch.where(stop, osc.coupling_stop, osc.coupling)
        self.osc_offset[:] = torch.where(stop, osc.offset_stop, osc.offset)
        self.stop_time = torch.where(stop[:, 0], self.stop_time + self.dt, 0.0)

    def _ramp_commands(self):
        """Move the forward-speed command toward its target at no more than
        commands.max_accel [m/s^2] (a joystick or planner ramps speed too)."""
        max_accel = getattr(self.cfg.commands, "max_accel", None)
        if max_accel is None:
            return
        step = max_accel * self.dt
        self.commands[:, 0] += torch.clamp(
            self.command_target - self.commands[:, 0], -step, step
        )

    def _resample_commands(self, command_mask):
        """Base sampling, then: commands.zero_prob of new commands are 'stop'
        (zero), forward commands below osc.stop_threshold also become stop,
        and a stop command has no yaw rate (no turning on the spot). With
        commands.max_accel the forward command then ramps to its new value,
        except right after a reset, where it starts at it. During the
        commands.startstop_ramp curriculum, stops come in gradually."""
        previous = self.commands[:, 0].clone()
        super()._resample_commands(command_mask)
        cmd = self.cfg.commands
        threshold = self.cfg.osc.stop_threshold
        share = self._startstop_scale()
        zero_prob = getattr(cmd, "zero_prob", 0.0) * share
        slow = torch.abs(self.commands[:, 0]) < threshold
        if share < 1.0:
            # Slow commands stop with probability `share`; the others are
            # raised to the stop threshold (walking at the slowest speed).
            keep = command_mask & slow & (
                torch.rand(self.num_envs, device=self.device) >= share
            )
            self.commands[keep, 0] = threshold
            slow &= ~keep
        stop = slow
        if zero_prob > 0.0:
            stop |= torch.rand(self.num_envs, device=self.device) < zero_prob
        stop &= command_mask
        self.commands[stop, 0] = 0.0
        self.commands[stop, 2] = 0.0
        if getattr(cmd, "max_accel", None) is not None:
            self.command_target[command_mask] = self.commands[command_mask, 0]
            ramp = command_mask & ~self._fresh
            self.commands[ramp, 0] = previous[ramp]

    def _compute_torques(self):
        torques = super()._compute_torques()
        if self._locked_dofs.numel() > 0:
            lock = self.cfg.leg_lock
            pos = self.dof_pos.index_select(1, self._locked_full_dofs)
            vel = self.dof_vel.index_select(1, self._locked_full_dofs)
            locked = lock.stiffness * (self._lock_pos - pos) - lock.damping * vel
            torques[:, self._locked_dofs] = torch.clip(
                locked, -self._lock_torque_limits, self._lock_torque_limits
            )
        return self._apply_motor_limits(torques)

    def _apply_motor_limits(self, torques):
        """Actuator limits (cfg.motor_limits), torque-only so every backend
        supports them. Speed (speed_scale x the velocity limit, the top speed):
        the torque available in the direction of motion falls linearly from
        full at taper_start x top speed to zero at the top speed (a motor's
        torque-speed curve), and above the top speed a braking torque of
        brake_gain [Nm per rad/s] x the overspeed pulls the joint back.
        Torque: every torque is then clipped to torque_scale x the effort
        limit. (A physics step is 10 ms, so a light joint can still overshoot
        the top speed briefly; the taper starts early to limit that.)"""
        limits = getattr(self.cfg, "motor_limits", None)
        if limits is None:
            return torques
        cap = self.actuated_torque_limits * limits.torque_scale
        if limits.speed_scale is not None:
            vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
            top = self.dof_vel_limits[self.actuated_dof_indices] * limits.speed_scale
            start = limits.taper_start * top
            room = ((top - vel.abs()) / (top - start)).clamp(0.0, 1.0)
            along = torques * vel > 0  # torque that speeds the joint up
            torques = torch.where(along, torques * room, torques)
            over = (vel.abs() - top).clamp(min=0.0)
            torques = torques - limits.brake_gain * over * torch.sign(vel)
        return torch.clip(torques, -cap, cap)

    def _post_physics_step(self):
        super()._post_physics_step()
        sim_dt = self.dt / self.cfg.control.decimation
        self._step_oscillators(sim_dt)
        self._update_load_stats(sim_dt)
        self._update_power_stats()

    def _update_power_stats(self):
        """Joint power and feet on the ground, averaged over the control
        step's physics substeps."""
        n = self.cfg.control.decimation
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        self.step_power += (self.torques * vel).abs().sum(dim=1) / n
        s = self.cfg.reward_settings
        down = ((self.grf > s.contact_threshold) & self.active_legs).sum(dim=1)
        self.step_flight += (down < s.flight_min_feet).float() / n

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.heading_target = wrap_to_pi(
            self.heading_target + self.commands[:, 2] * self.dt
        )
        alpha = torch.clamp(self.dt * self.osc_omega[:, 0], max=1.0)
        self.stride_vx += alpha * (self.base_lin_vel[:, 0] - self.stride_vx)
        self.stride_power += alpha * (self.step_power - self.stride_power)

    def _update_load_stats(self, sim_dt):
        """Per-foot load averaged over about one oscillator period (an
        exponential moving average with time constant 1 / omega), and the
        peak load of the current control step. Unclamped, unlike self.grf."""
        weight = 9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)
        load = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0) / weight
        alpha = torch.clamp(sim_dt * self.osc_omega, max=1.0)
        self.stride_load += alpha * (load - self.stride_load)
        torch.maximum(self.peak_load, load, out=self.peak_load)

    def _step_oscillators(self, dt):
        self.grf = self._foot_load()
        feedback = (
            self.osc_coupling
            * self._sensed_load()
            * (torch.cos(self.phase) + self.osc_offset)
        )
        noise = torch.randn_like(self.phase) * self.cfg.osc.process_noise_std
        self.phase_vel = 2 * torch.pi * (self.osc_omega - feedback + noise)
        self.phase = torch.remainder(self.phase + self.phase_vel * dt, 2 * torch.pi)
        self._update_phase_obs()

    def _update_phase_obs(self):
        self.phase_obs = torch.cat(
            (torch.cos(self.phase), torch.sin(self.phase)), dim=1
        ) * self.active_legs.repeat(2)

    def _sensed_load(self):
        """Foot load as the oscillators see it: delayed and noisy if
        configured, else the exact load (which the gait reward always uses)."""
        osc = self.cfg.osc
        if osc.grf_delay_max_steps == 0 and osc.grf_noise_std == 0.0:
            return self.grf
        self._load_history = torch.roll(self._load_history, 1, dims=1)
        self._load_history[:, 0] = self.grf
        load = self._load_history[
            torch.arange(self.num_envs, device=self.device), self._load_delay
        ]
        if osc.grf_noise_std > 0.0:
            load = load + torch.randn_like(load) * osc.grf_noise_std
        return torch.clamp(load, 0.0, 1.0)

    def _foot_load(self):
        """Vertical foot force normalized by robot weight, clamped to [0, 1]."""
        weight = 9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)
        return torch.clamp(
            self.contact_forces[:, self.feet_indices, 2] / weight, min=0.0, max=1.0
        )

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        phase = torch.rand_like(self.phase) * 2 * torch.pi
        masked_update(self.phase, phase, reset_mask)
        self.phase_vel.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        if self.cfg.osc.grf_delay_max_steps > 0:
            delay = torch.randint_like(
                self._load_delay, self.cfg.osc.grf_delay_max_steps + 1
            )
            self._load_delay[:] = torch.where(reset_mask, delay, self._load_delay)
            self._load_history[reset_mask] = 0.0
        omega_range = self.cfg.osc.omega_range
        if omega_range is not None:
            low, high = omega_range
            omega = low + (high - low) * torch.rand_like(self.episode_omega)
            masked_update(self.episode_omega, omega, reset_mask)
            masked_update(self.osc_omega, omega, reset_mask)
        # Start from an even split of the body weight over the active legs.
        even = self.active_legs / self.active_legs.sum()
        self.stride_load[reset_mask] = even
        self.peak_load[reset_mask] = 0.0
        self.heading_target = torch.where(
            reset_mask, yaw_of(self.root_states[:, 3:7]), self.heading_target
        )
        self.stride_vx.masked_fill_(reset_mask, 0.0)
        self.stride_power.masked_fill_(reset_mask, 0.0)
        self.stop_time.masked_fill_(reset_mask, 0.0)
        self._fresh |= reset_mask
        self._update_phase_obs()

    def reset_to_basic(self, reset_mask):
        super().reset_to_basic(reset_mask)
        self._reset_locked_leg(reset_mask)

    def reset_to_range(self, reset_mask):
        super().reset_to_range(reset_mask)
        self._reset_locked_leg(reset_mask)
        # init_state.rest_prob of resets start the base at rest.
        rest_prob = getattr(self.cfg.init_state, "rest_prob", 0.0)
        rest_prob *= self._startstop_scale()
        if rest_prob > 0.0:
            rest = reset_mask & (
                torch.rand(self.num_envs, device=self.device) < rest_prob
            )
            self.root_states[rest, 7:13] = 0.0

    def _reset_locked_leg(self, reset_mask):
        """Start the locked leg in its lock pose instead of snapping to it."""
        if self._locked_dofs.numel() == 0:
            return
        idx = self._locked_full_dofs
        self.dof_pos[:, idx] = torch.where(
            reset_mask.unsqueeze(1), self._lock_pos, self.dof_pos[:, idx]
        )
        self.dof_vel[:, idx] = torch.where(
            reset_mask.unsqueeze(1), 0.0, self.dof_vel[:, idx]
        )

    def _reward_gait(self):
        """Penalize load in swing [0, pi), reward load in stance [pi, 2 pi)."""
        return -torch.sum(self.grf * torch.sin(self.phase) * self.active_legs, dim=1)

    def _reward_even_load(self):
        """Penalize uneven load over the active legs, averaged over about a
        stride: minus the squared coefficient of variation of each leg's
        share (0 when even; n - 1 with all weight on one of n legs).
        Per-stride, so gaits whose legs land at different times are not
        penalized for it."""
        load = self.stride_load * self.active_legs
        share = load / load.sum(dim=1, keepdim=True).clamp(min=1e-6)
        n = self.active_legs.sum()
        deviation = (share - 1.0 / n) * self.active_legs
        return -n * torch.sum(deviation.square(), dim=1)

    def _reward_peak_load(self):
        """Penalize hard landings: squared excess of each active foot's peak
        load in this control step over reward_settings.peak_load_threshold
        [x body weight]."""
        excess = self.peak_load - self.cfg.reward_settings.peak_load_threshold
        return -torch.sum(excess.clamp(min=0.0).square() * self.active_legs, dim=1)

    def _reward_orientation(self):
        """Keep the body level, pitch and roll with their own widths
        (reward_settings.pitch_sigma, roll_sigma); equal to the base class's
        reward when both are its tracking_sigma."""
        s = self.cfg.reward_settings
        g = self.projected_gravity
        return torch.exp(-g[:, 0].square() / s.pitch_sigma) + torch.exp(
            -g[:, 1].square() / s.roll_sigma
        )

    def _reward_steady_speed(self):
        """Penalize forward-speed changes within a stride (the start-stop of
        a pronk or hop): squared deviation of the forward speed from its
        one-stride running average, in units of reward_settings.steady_sigma
        x the commanded speed (at least 0.5 m/s), capped at steady_cap. Off
        for steady_delay seconds after each reset and new command, so
        speeding up or slowing down to a new command is not penalized; the
        cap bounds the rest of that, and the large relative error of slow
        commands."""
        s = self.cfg.reward_settings
        scale = s.steady_sigma * self.commands[:, 0].abs().clamp(min=0.5)
        error = ((self.base_lin_vel[:, 0] - self.stride_vx) / scale).square()
        error = error.clamp(max=s.steady_cap)
        period = max(1, int(round(self.cfg.commands.resampling_time / self.dt)))
        since_command = (self.episode_length_buf % period) * self.dt
        return -error * (since_command >= s.steady_delay)

    def _push_scale(self):
        """Push curriculum: the push size grows linearly to its maximum over
        push_robots.ramp_steps control steps (0: full size from the start)."""
        ramp = getattr(self.cfg.push_robots, "ramp_steps", 0)
        return min(1.0, self.common_step_counter / ramp) if ramp else 1.0

    def _push_robots(self):
        """Base pushes at the curriculum's size. Scaled here rather than in
        the base class, so this file works with an unmodified base env."""
        cfg = self.cfg.push_robots
        full = cfg.max_push_vel_xy
        cfg.max_push_vel_xy = full * self._push_scale()
        try:
            super()._push_robots()
        finally:
            cfg.max_push_vel_xy = full

    def _startstop_scale(self):
        """Start/stop curriculum: 0 before commands.startstop_ramp[0] control
        steps, growing linearly to 1 at startstop_ramp[1] (None: always 1)."""
        ramp = getattr(self.cfg.commands, "startstop_ramp", None)
        if not ramp:
            return 1.0
        first, last = ramp
        if last <= first:
            return float(self.common_step_counter >= first)
        return min(1.0, max(0.0, (self.common_step_counter - first) / (last - first)))

    def _reward_power_smooth(self):
        """Penalize joint power that swings within a stride (all legs pushing
        at once, as in a hop): squared deviation of the total joint power from
        its one-stride running average, relative to that average (at least
        reward_settings.power_floor W), capped at power_smooth_cap. Off for
        power_smooth_delay seconds after a reset, while the average settles."""
        s = self.cfg.reward_settings
        mean = self.stride_power.clamp(min=s.power_floor)
        error = ((self.step_power - self.stride_power) / mean).square()
        error = error.clamp(max=s.power_smooth_cap)
        settled = self.episode_length_buf * self.dt >= s.power_smooth_delay
        return -error * settled

    def _reward_stand_still(self):
        """Penalize joint motion while commanded to stop: mean squared joint
        speed, phased in from reward_settings.stand_grace [s] after the stop
        command over the next stand_grace seconds, so braking is not
        penalized."""
        grace = self.cfg.reward_settings.stand_grace
        scale = torch.clamp((self.stop_time - grace) / grace, 0.0, 1.0)
        speed = self.dof_vel[:, self.actuated_dof_indices].square().mean(dim=1)
        return -speed * scale

    def _reward_joint_speed_limit(self):
        """Penalize joint speeds above reward_settings.speed_soft_limit of each
        joint's velocity limit: the excess as a share of the limit, summed
        over the actuated joints (not capped, so a joint far over the limit
        is pushed back harder)."""
        idx = self.actuated_dof_indices
        share = self.dof_vel[:, idx].abs() / self.dof_vel_limits[idx]
        excess = share - self.cfg.reward_settings.speed_soft_limit
        return -excess.clamp(min=0.0).sum(dim=1)

    def _reward_cost_of_transport(self):
        """Penalize energy per distance: joint power over weight x commanded
        speed (at least reward_settings.cot_min_speed), a dimensionless cost
        of transport (about 1.5 for the wide-study policies)."""
        s = self.cfg.reward_settings
        weight = 9.81 * self._backend.link_mass.sum(dim=1)
        speed = self.commands[:, 0].abs().clamp(min=s.cot_min_speed)
        return -self.step_power / (weight * speed)

    def _reward_flight(self):
        """At low commanded speed, penalize time with fewer than
        reward_settings.flight_min_feet active feet on the ground (flight
        phases and single-leg stances), so hopping and two-leg running are
        costly and walking gaits pay. Full below flight_speed[0], fading to
        zero at flight_speed[1] [m/s]."""
        low, high = self.cfg.reward_settings.flight_speed
        speed = self.commands[:, 0].abs()
        gate = ((high - speed) / (high - low)).clamp(0.0, 1.0)
        return -self.step_flight * gate

    def _reward_heading(self):
        """Keep the heading the robot had at reset (turned by any yaw-rate
        command), so it runs straight instead of drifting in a curve.
        Squared exponential with width reward_settings.heading_sigma [rad]."""
        error = wrap_to_pi(yaw_of(self.base_quat) - self.heading_target)
        return torch.exp(-error.square() / self.cfg.reward_settings.heading_sigma**2)
