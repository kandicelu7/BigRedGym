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
    # Start/stop state (deploy settings): robots reset since the last update,
    # and time the command has been 'stop'.
    task._fresh = torch.zeros(num_envs, dtype=torch.bool)
    task.stop_time = torch.zeros(num_envs)
    task.dt = 0.01  # control period [s]
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
    # Standing regime below the 0.2 m/s threshold (0.2 itself is moving).
    torch.testing.assert_close(
        task.osc_coupling[:, 0], torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])
    )
    torch.testing.assert_close(
        task.osc_offset[:, 0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
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


def test_even_load_is_zero_when_legs_share_equally_and_ignores_locked_leg():
    active = torch.tensor([True, False, True, True])
    task = make_task(num_envs=3, active_legs=active)
    task.stride_load = torch.tensor(
        [
            [0.3, 5.0, 0.3, 0.3],  # even over the active legs
            [0.9, 0.0, 0.0, 0.0],  # one leg carries everything
            [0.4, 0.0, 0.4, 0.1],
        ]
    )
    reward = task._reward_even_load()
    # Minus the squared coefficient of variation of the active legs' shares.
    shares = torch.tensor([0.4, 0.4, 0.1]) / 0.9
    cv2 = 3 * torch.sum((shares - 1 / 3) ** 2)
    torch.testing.assert_close(reward, torch.stack([torch.tensor(0.0), torch.tensor(-2.0), -cv2]))


def test_stride_load_averages_over_an_oscillator_period():
    task = make_task(foot_force=98.1)  # a load of 1 on every foot
    task.stride_load = torch.zeros(1, 4)
    task.peak_load = torch.zeros(1, 4)
    # omega = 2 Hz, so a time constant of 0.5 s: after 0.5 s, 1 - 1/e.
    for _ in range(500):
        task._update_load_stats(0.001)
    torch.testing.assert_close(
        task.stride_load, torch.full((1, 4), 1 - math.exp(-1)), atol=2e-3, rtol=0
    )


def test_peak_load_tracks_the_largest_substep_load_and_penalizes_its_excess():
    active = torch.tensor([True, False, True, True])
    task = make_task(active_legs=active)
    task.stride_load = torch.zeros(1, 4)
    task.peak_load = torch.zeros(1, 4)
    # Three physics substeps of one control step; FL lands hard in the second.
    for loads in ([0.5, 9.0, 0.8, 1.5], [2.5, 0.0, 0.3, 0.2], [1.2, 0.0, 0.1, 0.0]):
        task.contact_forces[:, :, 2] = 98.1 * torch.tensor(loads)
        task._update_load_stats(0.002)
    torch.testing.assert_close(task.peak_load, torch.tensor([[2.5, 9.0, 0.8, 1.5]]))
    # Threshold 1: (2.5 - 1)^2 + 0 + (1.5 - 1)^2; the locked leg is ignored.
    torch.testing.assert_close(task._reward_peak_load(), torch.tensor([-2.5]))


def test_heading_reward_is_one_on_course_and_falls_with_heading_error():
    task = make_task(num_envs=3)
    task.heading_target = torch.zeros(3)
    yaw = torch.tensor([0.0, 0.25, math.pi])
    task.base_quat = torch.stack(
        (torch.zeros(3), torch.zeros(3), torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=1
    )
    reward = task._reward_heading()
    torch.testing.assert_close(
        reward, torch.tensor([1.0, math.exp(-1.0), math.exp(-((math.pi / 0.25) ** 2))])
    )


def _base_sampling(monkeypatch, speed):
    """Replace the parent class's command sampling with a fixed forward speed."""

    def sample(self, mask):
        self.commands[mask, 0] = speed
        self.commands[mask, 2] = 0.3

    monkeypatch.setattr(Go2Osc.__mro__[1], "_resample_commands", sample)


def test_startstop_ramp_scale_grows_linearly_between_its_steps(monkeypatch):
    task = make_task()
    assert task._startstop_scale() == 1.0  # no ramp: full from the start
    monkeypatch.setattr(task.cfg.commands, "startstop_ramp", [100, 200])
    for step, share in [(0, 0.0), (100, 0.0), (150, 0.5), (200, 1.0), (500, 1.0)]:
        task.common_step_counter = step
        assert math.isclose(task._startstop_scale(), share)


def test_no_stop_commands_before_the_startstop_ramp(monkeypatch):
    _base_sampling(monkeypatch, speed=0.3)  # below the 0.5 m/s stop threshold
    task = make_task(num_envs=256)
    for name, value in [("zero_prob", 1.0), ("max_accel", None), ("startstop_ramp", [100, 200])]:
        monkeypatch.setattr(task.cfg.commands, name, value, raising=False)
    monkeypatch.setattr(task.cfg.osc, "stop_threshold", 0.5)
    task.commands = torch.zeros(256, 4)
    mask = torch.ones(256, dtype=torch.bool)

    task.common_step_counter = 0  # before the ramp: slow commands walk at 0.5
    task._resample_commands(mask)
    assert torch.all(task.commands[:, 0] == 0.5)
    assert torch.all(task.commands[:, 2] == 0.3)

    task.common_step_counter = 200  # after it: every command stops
    task._resample_commands(mask)
    assert torch.all(task.commands[:, 0] == 0.0)
    assert torch.all(task.commands[:, 2] == 0.0)


def test_pushes_grow_over_the_ramp_and_leave_the_config_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(
        Go2Osc.__mro__[1],
        "_push_robots",
        lambda self: seen.append(self.cfg.push_robots.max_push_vel_xy),
    )
    task = make_task()
    monkeypatch.setattr(task.cfg.push_robots, "max_push_vel_xy", 1.0)
    monkeypatch.setattr(task.cfg.push_robots, "ramp_steps", 100)
    for step in (25, 100, 400):
        task.common_step_counter = step
        task._push_robots()
    assert seen == [0.25, 1.0, 1.0]
    assert task.cfg.push_robots.max_push_vel_xy == 1.0


def test_joint_speed_penalty_counts_only_the_share_above_the_soft_limit(monkeypatch):
    task = make_task(num_envs=2)
    monkeypatch.setattr(task.cfg.reward_settings, "speed_soft_limit", 0.85)
    task.actuated_dof_indices = torch.arange(2)
    task.dof_vel_limits = torch.tensor([10.0, 20.0])
    # Robot 0 under the soft limit; robot 1: joint 0 at 105%, joint 1 at -95%.
    task.dof_vel = torch.tensor([[8.0, -16.0], [10.5, -19.0]])
    penalty = task._reward_joint_speed_limit()
    assert torch.allclose(penalty, torch.tensor([0.0, -(0.20 + 0.10)]))
