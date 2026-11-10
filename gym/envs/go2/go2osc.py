import torch

from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update


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
        self.osc_coupling = torch.zeros_like(self.osc_omega)
        self.osc_offset = torch.zeros_like(self.osc_omega)

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
        self._step_oscillators(self.dt / self.cfg.control.decimation)

    def _step_oscillators(self, dt):
        self.grf = self._foot_load()
        feedback = (
            self.osc_coupling * self.grf * (torch.cos(self.phase) + self.osc_offset)
        )
        noise = torch.randn_like(self.phase) * self.cfg.osc.process_noise_std
        self.phase_vel = 2 * torch.pi * (self.osc_omega - feedback + noise)
        self.phase = torch.remainder(self.phase + self.phase_vel * dt, 2 * torch.pi)
        self._update_phase_obs()

    def _update_phase_obs(self):
        self.phase_obs = torch.cat(
            (torch.cos(self.phase), torch.sin(self.phase)), dim=1
        ) * self.active_legs.repeat(2)

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
        omega_range = self.cfg.osc.omega_range
        if omega_range is not None:
            low, high = omega_range
            omega = low + (high - low) * torch.rand_like(self.episode_omega)
            masked_update(self.episode_omega, omega, reset_mask)
            masked_update(self.osc_omega, omega, reset_mask)
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
