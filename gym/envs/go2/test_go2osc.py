from types import SimpleNamespace

import torch

from gym.envs.go2.go2osc import Go2Osc, MID_SWING_PHASE

BODY_WEIGHT = 98.1
FR = 1


def make_task(num_envs=1, disabled_leg=FR, sigma=6.0, penalty=1.0, lift=2.0):
    task = Go2Osc.__new__(Go2Osc)
    task.num_envs = num_envs
    task.num_legs = 4
    task.device = "cpu"
    task.dt = 0.01
    task.cfg = SimpleNamespace(
        control=SimpleNamespace(decimation=1),
        osc=SimpleNamespace(
            coupling_sigma=sigma,
            disabled_leg_contact_penalty=penalty,
            disabled_leg_lift=lift,
        ),
    )
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(num_envs, 4, 3)
    task._backend = SimpleNamespace(link_mass=torch.full((num_envs, 1), 10.0))
    task.phase = torch.zeros(num_envs, 4)
    task.phase_obs = torch.zeros(num_envs, 8)
    task.phase_frequency = torch.full((num_envs, 1), 2.0)
    task.step_radius = torch.ones(num_envs, 4)
    task.disabled_leg_idx = torch.full((num_envs,), disabled_leg)
    task._leg_ids = torch.arange(4).unsqueeze(0)
    task._per_leg_buffers_ready = True
    task._gait_joint_offsets = torch.tensor(4 * [0.0, 0.96, -1.36])
    task._gait_joint_amplitudes = torch.tensor(4 * [0.0, -0.15, 0.30])
    task.gait_reference = torch.zeros(num_envs, 12)
    task._update_step_radius()
    return task


def load_leg(task, leg):
    task.contact_forces[:, leg, 2] = BODY_WEIGHT


def test_disabled_leg_has_zero_radius_and_others_one():
    task = make_task(disabled_leg=FR)
    torch.testing.assert_close(task.step_radius, torch.tensor([[1.0, 0.0, 1.0, 1.0]]))

    task = make_task(disabled_leg=-1)
    torch.testing.assert_close(task.step_radius, torch.ones(1, 4))


def test_step_grf_rewards_stance_load_and_penalizes_swing_load():
    task = make_task()
    load_leg(task, 0)

    task.phase[:, 0] = torch.pi / 2  # mid-stance
    torch.testing.assert_close(task._reward_step_grf(), torch.tensor([1.0]))

    task.phase[:, 0] = MID_SWING_PHASE
    torch.testing.assert_close(task._reward_step_grf(), torch.tensor([-1.0]))


def test_step_grf_penalizes_any_load_on_disabled_leg():
    task = make_task(penalty=0.5)
    task.phase[:, FR] = torch.pi / 2  # phase must not matter for a disabled leg
    torch.testing.assert_close(task._reward_step_grf(), torch.tensor([0.0]))

    load_leg(task, FR)
    torch.testing.assert_close(task._reward_step_grf(), torch.tensor([-0.5]))


def test_disabled_leg_phase_is_frozen_while_others_advance():
    task = make_task()
    task.phase[:] = MID_SWING_PHASE
    load_leg(task, FR)

    task._advance_phase()

    assert task.phase[0, FR] == MID_SWING_PHASE
    expected = MID_SWING_PHASE + 2 * torch.pi * 2.0 * 0.01
    torch.testing.assert_close(task.phase[0, 0], torch.tensor(expected))


def test_load_feedback_pulls_phase_toward_mid_stance():
    task = make_task(disabled_leg=-1)
    load_leg(task, 0)
    intrinsic = 2 * torch.pi * 2.0

    task.phase[:] = 0.0  # touchdown: a loaded leg hurries into stance
    assert task._phase_rate()[0, 0] > intrinsic
    torch.testing.assert_close(task._phase_rate()[0, 2], torch.tensor(intrinsic))

    task.phase[:] = torch.pi  # lift-off: a loaded leg is held back
    assert task._phase_rate()[0, 0] < intrinsic


def test_gait_reference_is_per_leg_and_static_for_disabled_leg():
    task = make_task(lift=2.0)
    task.phase[:] = torch.tensor([[torch.pi / 2, 0.3, MID_SWING_PHASE, 0.0]])

    task._update_gait_reference()
    reference = task.gait_reference.view(4, 3)

    torch.testing.assert_close(reference[0], torch.tensor([0.0, 0.81, -1.06]))
    torch.testing.assert_close(reference[2], torch.tensor([0.0, 1.11, -1.66]))
    torch.testing.assert_close(reference[3], torch.tensor([0.0, 0.96, -1.36]))
    # Disabled leg: lifted twice the swing amplitude, whatever its phase.
    torch.testing.assert_close(reference[FR], torch.tensor([0.0, 1.26, -1.96]))


def test_phase_observation_is_sin_then_cos_per_leg():
    task = make_task()
    task.phase[:] = torch.tensor([[0.0, torch.pi / 2, torch.pi, MID_SWING_PHASE]])

    task._update_phase_observation()

    torch.testing.assert_close(
        task.phase_obs,
        torch.tensor([[0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0]]),
        atol=1e-6,
        rtol=0,
    )
