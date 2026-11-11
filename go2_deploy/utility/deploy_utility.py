import torch
from go2_deploy.deploy_config import DeployConfig
from gym.utils.torch_quat import quat_rotate_inverse

# Unitree motor order: Front Right hip (haa), FR thigh (hfe), FR calf (kfe),
# Front Left ... Rear Right ... Rear Left
# QGym motor order: FL hip, FL thigh, FL calf, FR ... RL ... RR
UNITREE_TO_QGYM_JOINT_IDX = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
QGYM_TO_UNITREE_JOINT_IDX = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

# Compute pieces of the observation vector, returning torch tensors


def _get_obs_base_ang_vel(main_controller, lowstate_msg):
    return torch.tensor(lowstate_msg.imu_state.gyroscope)


def _get_obs_projected_gravity(main_controller, lowstate_msg):
    # Convert Unitree WXYZ to QGym XYZW
    base_quat = torch.tensor(lowstate_msg.imu_state.quaternion)[[1, 2, 3, 0]]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0])
    return quat_rotate_inverse(base_quat, gravity_vec)


def _get_obs_commands(main_controller, lowstate_msg):
    return main_controller.last_command


def _get_obs_dof_pos_obs(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    dof_pos_unitree_convention = torch.zeros(12)
    for i in range(12):
        dof_pos_unitree_convention[i] = motor_states[i].q
    return (
        dof_pos_unitree_convention[UNITREE_TO_QGYM_JOINT_IDX]
        - main_controller.cfg.default_dof_pos
    )


def _get_obs_dof_vel(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    dof_vel_unitree_convention = torch.zeros(12)
    for i in range(12):
        dof_vel_unitree_convention[i] = motor_states[i].dq
    return dof_vel_unitree_convention[UNITREE_TO_QGYM_JOINT_IDX]


def _get_obs_dof_accel(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    dof_accel_unitree_convention = torch.zeros(12)
    for i in range(12):
        dof_accel_unitree_convention[i] = motor_states[i].ddq
    return dof_accel_unitree_convention[UNITREE_TO_QGYM_JOINT_IDX]


def _get_obs_dof_pos_target(main_controller, lowstate_msg):
    # Previous applied policy offset in radians, after optional action smoothing.
    return main_controller.last_action[0]


def _get_obs_phase_obs(main_controller, lowstate_msg):
    phase = torch.tensor([main_controller.phase])
    return torch.tensor([torch.sin(phase), torch.cos(phase)])


def _get_obs_phase_frequency(main_controller, lowstate_msg):
    return torch.tensor([main_controller.phase_frequency])


# Returns torch tensor: observation vector from lowstate_msg
def lowstate_to_obs(main_controller, lowstate_msg):
    get_obs_piece = {
        "base_ang_vel": _get_obs_base_ang_vel,
        "projected_gravity": _get_obs_projected_gravity,
        "commands": _get_obs_commands,
        "dof_pos_obs": _get_obs_dof_pos_obs,
        "dof_vel": _get_obs_dof_vel,
        "dof_accel": _get_obs_dof_accel,
        "dof_pos_target": _get_obs_dof_pos_target,
        "phase_obs": _get_obs_phase_obs,
        "phase_frequency": _get_obs_phase_frequency,
    }

    obs_vector = torch.zeros(main_controller.obs_vec_size)
    i = 0
    for obs in main_controller.cfg.obs_vector:
        obs_size = main_controller.cfg.obs_sizes[obs]
        scale = torch.tensor(getattr(main_controller.cfg.DeployScaling, obs, 1.0))
        obs_vector[i : i + obs_size] = (
            get_obs_piece[obs](main_controller, lowstate_msg) / scale
        )
        i += obs_size

    return obs_vector


# Return torch.tensor(12): actual target position from Actor output
def action_to_target_pos(main_controller, action):
    return action + main_controller.cfg.default_dof_pos


# Return torch.tensor(12): the Actor output corresponding to
# absolute position target_pos
def target_pos_to_action(main_controller, target_pos):
    return target_pos - main_controller.cfg.default_dof_pos


# Returns LowCmd_ from the actor output action_qgm_convention, kp_mult
# Adds the static default posture and maps targets to Unitree motor order.
def action_to_lowcmd(main_controller, action_qgym_convention, kp_mult=1.0, kd_mult=1.0):
    target_pos_qgym = action_to_target_pos(main_controller, action_qgym_convention)
    target_pos = target_pos_qgym[QGYM_TO_UNITREE_JOINT_IDX]

    lowcmd = main_controller.default_lowcmd
    for i in range(12):
        lowcmd.motor_cmd[i].q = target_pos[i].item()
        lowcmd.motor_cmd[i].kp = main_controller.cfg.kp * kp_mult
        lowcmd.motor_cmd[i].kd = main_controller.cfg.kd * kd_mult

    return lowcmd


# Returns LowCmd_ of q, dq, kp, kd = 0
def default_lowcmd():
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    lowcmd = unitree_go_msg_dds__LowCmd_()
    crc = CRC()
    lowcmd.head[0] = 0xFE
    lowcmd.head[1] = 0xEF
    lowcmd.level_flag = 0xFF
    lowcmd.gpio = 0
    cfg = DeployConfig()

    for i in range(20):
        lowcmd.motor_cmd[i].mode = 0x01
        lowcmd.motor_cmd[i].q = 0
        lowcmd.motor_cmd[i].kp = 0
        lowcmd.motor_cmd[i].dq = 0
        lowcmd.motor_cmd[i].kd = 0
        lowcmd.motor_cmd[i].tau = 0

    # the motors that are actually used
    for i in range(12):
        lowcmd.motor_cmd[i].kp = cfg.kp
        lowcmd.motor_cmd[i].kd = cfg.kd

    lowcmd.crc = crc.Crc(lowcmd)

    return lowcmd


# Returns LowCmd_ of q, dq, kp = 0, kd > 0
def emergency_lowcmd():
    from unitree_sdk2py.utils.crc import CRC

    lowcmd = default_lowcmd()
    crc = CRC()

    for i in range(12):
        lowcmd.motor_cmd[i].kp = 0
        lowcmd.motor_cmd[i].kd = 5.0

    lowcmd.crc = crc.Crc(lowcmd)

    return lowcmd
