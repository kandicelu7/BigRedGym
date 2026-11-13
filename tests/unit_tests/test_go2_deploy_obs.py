"""Current deployment utilities, independent of the Unitree SDK/DDS stack.

These tests check deployment observation and action utilities with synthetic
messages; they do not compare deployment against a simulated task.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility

SDK_IN_QGYM_ORDER = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def _controller(cfg=None):
    if cfg is None:
        cfg = DeployConfig()
    return SimpleNamespace(
        cfg=cfg,
        obs_vec_size=sum(cfg.obs_sizes[name] for name in cfg.obs_vector),
        last_command=torch.tensor([0.5, -0.25, 0.75]),
        last_action=torch.zeros(2, 12),
        phase=0.4,
        phase_frequency=cfg.phase_frequency,
    )


def _message():
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=float(i), dq=-2.0 * i, ddq=3.0 * i) for i in range(12)
        ],
        imu_state=SimpleNamespace(
            quaternion=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
            gyroscope=[0.1, -0.2, 0.3],
        ),
    )


def test_joint_mapping_for_positions_velocities_and_accelerations():
    controller, message = _controller(), _message()
    controller.cfg.default_dof_pos = torch.linspace(0.1, 1.2, 12)
    expected = torch.tensor(SDK_IN_QGYM_ORDER, dtype=torch.float)
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_pos_obs(controller, message),
        expected - controller.cfg.default_dof_pos,
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_vel(controller, message), -2 * expected
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_accel(controller, message), 3 * expected
    )
    canonical = torch.arange(12)
    sdk = canonical[deploy_utility.QGYM_TO_UNITREE_JOINT_IDX]
    assert sdk.tolist() == SDK_IN_QGYM_ORDER
    assert torch.equal(sdk[deploy_utility.UNITREE_TO_QGYM_JOINT_IDX], canonical)


def test_zero_action_and_nominal_posture_have_zero_relative_observations():
    controller, message = _controller(), _message()
    nominal = torch.tensor([0.0, 0.96, -1.36] * 4)
    torch.testing.assert_close(
        deploy_utility.action_to_target_pos(controller, torch.zeros(12)), nominal
    )
    for motor, position in zip(message.motor_state, nominal[SDK_IN_QGYM_ORDER]):
        motor.q = position.item()
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_pos_obs(controller, message), torch.zeros(12)
    )


def test_wxyz_quaternion_projects_gravity_but_gyro_is_already_body_local():
    controller, message = _controller(), _message()
    # A positive quarter-turn about X maps world-down to body negative Y.
    torch.testing.assert_close(
        deploy_utility._get_obs_projected_gravity(controller, message),
        torch.tensor([0.0, -1.0, 0.0]),
        atol=2e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_base_ang_vel(controller, message),
        torch.tensor([0.1, -0.2, 0.3]),
    )


def test_observation_dispatch_preserves_configured_order_sizes_and_scaling():
    cfg = DeployConfig()
    cfg.obs_vector = [
        "phase_frequency",
        "dof_vel",
        "base_ang_vel",
        "dof_pos_target",
        "commands",
        "projected_gravity",
        "dof_pos_obs",
        "dof_accel",
        "phase_obs",
    ]
    cfg.DeployScaling = SimpleNamespace(
        phase_frequency=4.0,
        dof_vel=2.0,
        base_ang_vel=0.5,
        dof_pos_target=0.25,
        commands=[2.0, 4.0, 8.0],
        projected_gravity=2.0,
        dof_pos_obs=list(range(1, 13)),
        dof_accel=3.0,
        phase_obs=2.0,
    )
    controller = _controller(cfg)
    controller.last_action[0] = torch.linspace(-6.0, 5.0, 12)
    ordered = torch.tensor(SDK_IN_QGYM_ORDER, dtype=torch.float)
    expected = torch.cat(
        (
            torch.tensor([controller.phase_frequency / 4]),
            -ordered,
            torch.tensor([0.2, -0.4, 0.6]),
            controller.last_action[0] / 0.25,
            torch.tensor([0.25, -0.0625, 0.09375]),
            torch.tensor([0.0, -0.5, 0.0]),
            (ordered - cfg.default_dof_pos) / torch.arange(1, 13),
            ordered,
            torch.tensor([math.sin(0.4), math.cos(0.4)]) / 2,
        )
    )

    actual = deploy_utility.lowstate_to_obs(controller, _message())

    assert actual.shape == (controller.obs_vec_size,)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)


def test_frequency_observation_tracks_active_controller_frequency():
    cfg = DeployConfig()
    cfg.phase_frequency = 1.25
    cfg.obs_vector = ["phase_frequency"]
    cfg.DeployScaling = SimpleNamespace(phase_frequency=4.0)
    controller = _controller(cfg)

    # The oscillator's current frequency can differ from both config defaults.
    for frequency in (3.0, 4.5):
        controller.phase_frequency = frequency
        torch.testing.assert_close(
            deploy_utility.lowstate_to_obs(controller, _message()),
            torch.tensor([frequency / 4.0]),
        )


@pytest.mark.parametrize("task_name", ["go2trot", "go2"])
def test_action_target_roundtrip_uses_static_default_pose(task_name):
    controller = _controller()
    controller.cfg.task_name = task_name
    controller.cfg.default_dof_pos = torch.linspace(0.1, 1.2, 12)
    action = torch.linspace(-0.2, 0.2, 12)

    target = deploy_utility.action_to_target_pos(controller, action)

    torch.testing.assert_close(
        target,
        action + controller.cfg.default_dof_pos,
    )
    torch.testing.assert_close(
        deploy_utility.target_pos_to_action(controller, target), action
    )
    controller.phase += math.pi
    torch.testing.assert_close(
        deploy_utility.action_to_target_pos(controller, action), target
    )


def test_existing_lowcmd_receives_permuted_unrestricted_targets_and_scaled_gains():
    controller = _controller()
    controller.default_lowcmd = SimpleNamespace(
        motor_cmd=[SimpleNamespace(q=0.0, kp=0.0, kd=0.0) for _ in range(20)]
    )
    canonical_target = torch.arange(12, dtype=torch.float) - 6
    action = canonical_target - controller.cfg.default_dof_pos

    command = deploy_utility.action_to_lowcmd(
        controller, action, kp_mult=0.5, kd_mult=0.25
    )

    expected = canonical_target[SDK_IN_QGYM_ORDER]
    assert command is controller.default_lowcmd
    torch.testing.assert_close(
        torch.tensor([motor.q for motor in command.motor_cmd[:12]]), expected
    )
    assert all(motor.kp == controller.cfg.kp * 0.5 for motor in command.motor_cmd[:12])
    assert all(motor.kd == controller.cfg.kd * 0.25 for motor in command.motor_cmd[:12])
    assert all(motor.q == 0.0 for motor in command.motor_cmd[12:])
