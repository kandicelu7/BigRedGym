"""A standing robot's world contact force must not depend on its yaw."""

import math

import pytest
import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.base.vsim_backend import VSimBackend
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.task_registry import task_registry
from tests.unit_tests.conftest import vsim_guard


@pytest.mark.vsim
def test_rotated_standing_go2_contacts_balance_gravity():
    vsim_guard()
    cfg = Go2TrotCfg()
    cfg.seed = 123
    apply_domain_randomization_override(cfg, "off")
    task_registry.convert_frequencies_to_params(cfg, Go2TrotRunnerCfg())
    backend = VSimBackend()
    try:
        backend.setup(cfg, num_envs=4, device="cuda:0", task=None)
        # A legal, symmetric stance. Each independent world uses a different
        # yaw so a mistaken sensor-frame rotation cannot pass at identity only.
        target = torch.tensor([0.0, 0.66, -1.36] * 4, device="cuda:0")
        yaw = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2], device="cuda:0")
        backend.root_states.zero_()
        backend.root_states[:, 2] = 0.35
        backend.root_states[:, 5] = torch.sin(yaw / 2)
        backend.root_states[:, 6] = torch.cos(yaw / 2)
        backend.dof_pos[:] = target
        backend.dof_vel.zero_()
        backend.reset_state(torch.ones(4, dtype=torch.bool, device="cuda:0"))
        total = torch.zeros(4, 3, device="cuda:0")
        for step in range(700):
            torque = 20 * (target - backend.dof_pos) - 0.5 * backend.dof_vel
            backend.step(torque.clamp(-20, 20))
            if step >= 500:
                total += backend.contact_forces.sum(dim=1)
        relative = total / 200 / (9.81 * backend.link_mass.sum(dim=1, keepdim=True))
        assert backend.root_states[:, 7:10].norm(dim=1).max() < 0.05
        torch.testing.assert_close(
            relative,
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0").expand(4, -1),
            atol=0.03,
            rtol=0,
        )
    finally:
        backend.close()
