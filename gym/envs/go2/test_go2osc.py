import math
from types import SimpleNamespace

import torch

from gym.envs.go2.go2osc import Go2Osc
from gym.envs.go2.go2osc_config import Go2OscCfg


def make_task(num_envs=1, foot_force=0.0, phase=0.0, active_legs=None):
    task = Go2Osc.__new__(Go2Osc)
    task.cfg = Go2OscCfg()
    task.cfg.osc.process_noise_std = 0.0
    task.device = "cpu"
    task.num_envs = num_envs
    task.feet_indices = torch.arange(4)
    # Total mass 10 kg, so a 98.1 N vertical force is a load of 1.
    task._backend = SimpleNamespace(link_mass=torch.full((num_envs, 2), 5.0))
    task.contact_forces = torch.zeros(num_envs, 4, 3)
    task.contact_forces[:, :, 2] = foot_force
    task.active_legs = (
        torch.ones(4, dtype=torch.bool) if active_legs is None else active_legs
    )
    task.phase = torch.full((num_envs, 4), phase)
    task.osc_omega = torch.full((num_envs, 1), 2.0)
    task.osc_coupling = torch.ones(num_envs, 1)
    task.osc_offset = torch.zeros(num_envs, 1)
    return task


def test_unloaded_oscillator_advances_at_nominal_frequency():
    task = make_task()
    task._step_oscillators(0.01)
    torch.testing.assert_close(task.phase, torch.full((1, 4), 2 * math.pi * 2.0 * 0.01))


def test_load_slows_phase_near_liftoff_and_speeds_it_after_touchdown():
    # phi = 0 is the end of stance: cos = 1, so load delays liftoff.
    task = make_task(foot_force=98.1, phase=0.0)
    task._step_oscillators(0.01)
    torch.testing.assert_close(task.phase_vel, torch.full((1, 4), 2 * math.pi * 1.0))

    # phi = pi is touchdown: cos = -1, so load speeds the phase into stance.
    task = make_task(foot_force=98.1, phase=math.pi)
    task._step_oscillators(0.01)
    torch.testing.assert_close(task.phase_vel, torch.full((1, 4), 2 * math.pi * 3.0))


def test_osc_params_follow_commanded_speed():
    task = make_task(num_envs=5)
    task.commands = torch.zeros(5, 3)
    task.commands[:, 0] = torch.tensor([0.0, 0.2, 1.0, 2.0, 3.0])
    task._update_osc_params()
    # Frequency is continuous in speed, capped at omega_max.
    torch.testing.assert_close(
        task.osc_omega[:, 0], torch.tensor([1.0, 1.25, 2.25, 3.5, 4.0])
    )
    # Standing regime up to and including the 0.2 m/s threshold.
    torch.testing.assert_close(
        task.osc_coupling[:, 0], torch.tensor([4.0, 4.0, 1.0, 1.0, 1.0])
    )
    torch.testing.assert_close(
        task.osc_offset[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    )


def test_locked_leg_is_masked_from_observations_and_gait_reward():
    active = torch.tensor([True, False, True, True])
    task = make_task(foot_force=98.1, phase=1.5 * math.pi, active_legs=active)
    task._step_oscillators(0.0)

    assert torch.all(task.phase_obs[:, [1, 5]] == 0.0)
    assert torch.all(task.phase_obs[:, [0, 2, 3, 4, 6, 7]] != 0.0)
    # Full load in mid-stance on the three active legs.
    torch.testing.assert_close(task._reward_gait(), torch.tensor([3.0]))


def test_omega_range_holds_episode_frequency_regardless_of_speed(monkeypatch):
    task = make_task(num_envs=3)
    monkeypatch.setattr(task.cfg.osc, "omega_range", [1.25, 4.0])
    task.commands = torch.zeros(3, 3)
    task.commands[:, 0] = torch.tensor([0.5, 1.5, 2.5])
    task.episode_omega = torch.tensor([[1.5], [3.0], [3.75]])
    task._update_osc_params()
    torch.testing.assert_close(task.osc_omega, task.episode_omega)
