import torch

from gym.utils.sampling import torch_rand_float
from gym.utils.sampling import masked_update

from gym.envs.go2.go2trot import Go2Trot

# Phase convention (shared with Go2Trot): sin(phase) > 0 is stance, with
# mid-stance at pi/2 and mid-swing at 3*pi/2.
MID_SWING_PHASE = 1.5 * torch.pi


class Go2Osc(Go2Trot):
    """Go2 with one load-feedback oscillator per leg and no prescribed gait.

    Each leg i has its own phase. Nothing couples the phases directly; they
    interact only through the ground forces the legs share (Owaki & Ishiguro,
    Sci. Rep. 2017):

        d(phase_i)/dt = radius_i * (2*pi*frequency + sigma * load_i * cos(phase_i))

    ``radius_i`` is the commanded step size of leg i. A leg with radius 0 is
    disabled: its phase is frozen at mid-swing and its joint reference is a
    static lifted pose.
    """

    def _init_buffers(self):
        # Go2Trot._init_buffers calls the phase/reference update hooks before
        # the per-leg buffers exist; the hooks fall back to Go2Trot until then.
        self._per_leg_buffers_ready = False
        super()._init_buffers()

        num_legs = len(self.robot_layout.body_groups["feet"])
        self.num_legs = num_legs
        self.phase = torch.zeros(
            self.num_envs, num_legs, dtype=torch.float, device=self.device
        )
        self.phase_obs = torch.zeros(
            self.num_envs, 2 * num_legs, dtype=torch.float, device=self.device
        )
        self.step_radius = torch.ones(
            self.num_envs, num_legs, dtype=torch.float, device=self.device
        )
        # Index of each environment's disabled leg, or -1 for none. Resets
        # derive step_radius from this, so evaluation scripts can overwrite it.
        self.disabled_leg_idx = torch.full(
            (self.num_envs,),
            self._configured_disabled_leg(),
            dtype=torch.long,
            device=self.device,
        )
        self._leg_ids = torch.arange(num_legs, device=self.device).unsqueeze(0)
        self._per_leg_buffers_ready = True

        self._update_step_radius()
        self._update_phase_observation()
        self._update_gait_reference()

    def _configured_disabled_leg(self):
        name = self.cfg.osc.disabled_leg
        if name is None:
            return -1
        foot_names = list(self.robot_layout.body_groups["feet"])
        if name not in foot_names:
            raise ValueError(
                f"cfg.osc.disabled_leg={name!r} is not one of {foot_names} (or None)"
            )
        return foot_names.index(name)

    def _update_step_radius(self):
        disabled = self._leg_ids == self.disabled_leg_idx.unsqueeze(1)
        self.step_radius[:] = (~disabled).float()

    def _update_phase_observation(self):
        if not self._per_leg_buffers_ready:
            return super()._update_phase_observation()
        self.phase_obs[:] = torch.cat(
            (torch.sin(self.phase), torch.cos(self.phase)), dim=1
        )

    def _update_gait_reference(self):
        if not self._per_leg_buffers_ready:
            return super()._update_gait_reference()
        # Active leg: radius scales the step about the nominal posture.
        # Disabled leg (radius 0): static pose, lifted past the swing apex.
        stride = (
            self.step_radius * torch.sin(self.phase)
            - (1.0 - self.step_radius) * self.cfg.osc.disabled_leg_lift
        )
        self.gait_reference[:] = (
            self._gait_joint_offsets
            + self._gait_joint_amplitudes * stride.repeat_interleave(3, dim=1)
        )

    def _leg_phases(self):
        return self.phase

    def _phase_rate(self):
        """Per-leg phase velocity [rad/s] with local load feedback."""
        intrinsic = 2 * torch.pi * self.phase_frequency
        feedback = (
            self.cfg.osc.coupling_sigma
            * self._foot_contact_strength()
            * torch.cos(self.phase)
        )
        return self.step_radius * (intrinsic + feedback)

    def _advance_phase(self):
        # Called once per physics substep.
        substep_dt = self.dt / self.cfg.control.decimation
        self.phase.add_(self._phase_rate() * substep_dt).remainder_(2 * torch.pi)

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        self._update_step_radius()
        # Independent random phases: no initial inter-leg relationship.
        phase = torch_rand_float(
            0, 2 * torch.pi, shape=self.phase.shape, device=self.device
        )
        phase = torch.where(
            self.step_radius > 0, phase, torch.full_like(phase, MID_SWING_PHASE)
        )
        masked_update(self.phase, phase, reset_mask)

    def _reward_step_grf(self):
        """Load in stance is rewarded, load in swing is penalized, scaled by
        step radius. Any load on a disabled leg (radius 0) is penalized."""
        load = self._foot_contact_strength()
        per_leg = load * (
            self.step_radius * torch.sin(self.phase)
            - self.cfg.osc.disabled_leg_contact_penalty * (1.0 - self.step_radius)
        )
        return per_leg.sum(dim=1)

    def _reward_disabled_leg_still(self):
        """Penalize joint motion of disabled legs."""
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        vel = vel / self.scales["dof_vel"]
        leg_motion = vel.square().view(self.num_envs, self.num_legs, -1).sum(dim=2)
        return -((1.0 - self.step_radius) * leg_motion).sum(dim=1)
