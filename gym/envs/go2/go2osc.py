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
        self._load_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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

    def _init_leg_lock(self):
        lock = self.cfg.leg_lock
        foot_names = self.robot_layout.body_groups["feet"]
        if lock.leg is not None and not any(
            name.startswith(lock.leg + "_") for name in foot_names
        ):
            raise ValueError(f"unknown leg_lock.leg {lock.leg!r}")
        # The locked leg's oscillator is meaningless; mask it out.
        self.active_legs = torch.tensor(
            [lock.leg is None or not name.startswith(lock.leg + "_")
             for name in foot_names],
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
        # The policy has no authority over the locked leg; zero its targets so
        # observations and action-rate penalties ignore it.
        self.dof_pos_target[:, self._locked_dofs] = 0.0
        self._update_osc_params()

    def _update_osc_params(self):
        """Oscillator parameters from the commanded forward speed.

        Frequency grows continuously with speed; coupling and offset switch to
        the standing regime of Eq. 5 at low speed.
        """
        osc = self.cfg.osc
        speed = torch.abs(self.commands[:, 0:1])
        if osc.omega_range is not None:
            self.osc_omega[:] = self.episode_omega
        else:
            self.osc_omega[:] = torch.clamp(
                osc.omega_min + osc.omega_slope * speed, max=osc.omega_max
            )
        stop = speed <= osc.stop_threshold
        self.osc_coupling[:] = torch.where(stop, osc.coupling_stop, osc.coupling)
        self.osc_offset[:] = torch.where(stop, osc.offset_stop, osc.offset)

    def _compute_torques(self):
        torques = super()._compute_torques()
        if self._locked_dofs.numel() == 0:
            return torques
        lock = self.cfg.leg_lock
        pos = self.dof_pos.index_select(1, self._locked_full_dofs)
        vel = self.dof_vel.index_select(1, self._locked_full_dofs)
        locked = lock.stiffness * (self._lock_pos - pos) - lock.damping * vel
        torques[:, self._locked_dofs] = torch.clip(
            locked, -self._lock_torque_limits, self._lock_torque_limits
        )
        return torques

    def _post_physics_step(self):
        super()._post_physics_step()
        sim_dt = self.dt / self.cfg.control.decimation
        self._step_oscillators(sim_dt)
        self._update_load_stats(sim_dt)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.heading_target = wrap_to_pi(
            self.heading_target + self.commands[:, 2] * self.dt
        )

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
        load = self._load_history[torch.arange(self.num_envs, device=self.device), self._load_delay]
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
            delay = torch.randint_like(self._load_delay, self.cfg.osc.grf_delay_max_steps + 1)
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
        self._update_phase_obs()

    def reset_to_basic(self, reset_mask):
        super().reset_to_basic(reset_mask)
        self._reset_locked_leg(reset_mask)

    def reset_to_range(self, reset_mask):
        super().reset_to_range(reset_mask)
        self._reset_locked_leg(reset_mask)

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
        return -torch.sum(
            self.grf * torch.sin(self.phase) * self.active_legs, dim=1
        )

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

    def _reward_heading(self):
        """Keep the heading the robot had at reset (turned by any yaw-rate
        command), so it runs straight instead of drifting in a curve.
        Squared exponential with width reward_settings.heading_sigma [rad]."""
        error = wrap_to_pi(yaw_of(self.base_quat) - self.heading_target)
        return torch.exp(-error.square() / self.cfg.reward_settings.heading_sigma**2)
