"""Plain MuJoCo (mj_step) backend — no Warp dependency.

Works on any platform (Linux CPU, Mac Apple Silicon). Each environment owns an
MjData. Models are shared unless link-mass DR requires one compact model per
environment; physics runs in a Python loop and state is copied numpy→torch
after each step.
"""

import copy
import time

import mujoco
import numpy as np
import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.domain_randomization import (
    contact_friction_range,
    link_mass_scale_range,
)
from gym.envs.base.mujoco_backend_base import (
    MuJocoBackendBase,
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
)
from gym.utils.torch_quat import quat_apply, quat_rotate_inverse


class MuJocoCPUBackend(MuJocoBackendBase):
    """SimBackend backed by plain mujoco.mj_step."""

    def __init__(self) -> None:
        super().__init__()
        self._datas: list[mujoco.MjData] = []

        # State tensors (allocated in setup)
        self._dof_state_t: torch.Tensor = None  # [N, num_dof, 2]
        self._dof_pos_view: torch.Tensor = None  # [N, num_dof] view into above
        self._dof_vel_view: torch.Tensor = None  # [N, num_dof] view into above
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, num_bodies, 13]
        self._contact_forces_t: torch.Tensor = None  # [N, num_bodies, 3]

        # Viewer (created lazily on first render call)
        self._viewer = None
        self._show_ui = False  # overridden from cfg.viewer.show_ui in setup()
        self._viewer_key_callback = None
        self._viewer_overlay_fn = None  # called each render() before sync()
        self._render_deadline = None
        self._render_pending_time = 0.0
        self._randomize_contact_friction = False
        self._randomize_link_mass = False
        self._models: list[mujoco.MjModel] | None = None

    # ── State tensors ──────────────────────────────────────────────────────────

    @property
    def dof_state(self) -> torch.Tensor:
        return self._dof_state_t.view(self._num_envs * self._num_dof, 2)

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos_view

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel_view

    @property
    def root_states(self) -> torch.Tensor:
        return self._root_states_t

    @property
    def rigid_body_states(self) -> torch.Tensor:
        return self._rigid_body_states_t.view(self._num_envs * self._num_bodies, 13)

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    @property
    def contact_friction(self) -> torch.Tensor:
        return self._contact_friction_t

    def set_contact_friction(
        self,
        env_ids: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        ids, values = self._prepare_contact_friction_update(env_ids, coefficients)
        if ids.numel() == 0:
            return
        if not self._randomize_contact_friction:
            raise RuntimeError(
                "contact-friction randomization was not enabled before setup"
            )
        self._contact_friction_t[ids] = values
        if self._models is not None:
            for env_id, value in zip(ids.tolist(), values.tolist()):
                self._set_model_contact_friction(self._models[env_id], value)

    def set_link_mass_scale(
        self,
        env_ids: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        ids, values = self._prepare_link_mass_scale_update(env_ids, scales)
        if ids.numel() == 0:
            return
        if not self._randomize_link_mass:
            raise RuntimeError("link-mass randomization was not enabled before setup")

        masses = self._nominal_link_mass * values
        inertias = self._nominal_link_inertia * values.unsqueeze(-1)
        self._link_mass_t[ids] = masses
        self._link_inertia_t[ids] = inertias
        body_indices = self._canonical_to_native_body_np
        for row, env_id in enumerate(ids.tolist()):
            model = self._models[env_id]
            data = self._datas[env_id]
            model.body_mass[body_indices] = masses[row].cpu().numpy()
            model.body_inertia[body_indices] = inertias[row].cpu().numpy()

            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            warmstart = data.qacc_warmstart.copy()
            time = data.time
            mujoco.mj_setConst(model, data)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.time = time
            mujoco.mj_forward(model, data)
            data.qacc_warmstart[:] = warmstart

    def _activate_domain(self, env_id: int) -> None:
        """Activate one environment's parameters on the shared CPU model."""
        if not self._randomize_contact_friction or self._models is not None:
            return
        self._set_model_contact_friction(
            self._mjm, float(self._contact_friction_t[env_id])
        )

    def _model_for_env(self, env_id: int) -> mujoco.MjModel:
        if self._models is None:
            return self._mjm
        return self._models[env_id]

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        self._device = device
        self._num_envs = num_envs
        self._randomize_contact_friction = contact_friction_range(cfg) is not None
        self._randomize_link_mass = link_mass_scale_range(cfg) is not None
        headless = task.headless if task is not None else True
        visual_model = getattr(cfg.asset, "mujoco_visual_model", None)
        visual_model_path = (
            visual_model.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
            if visual_model is not None and not headless
            else None
        )

        mjm = self._load_model(
            cfg,
            discard_visual=self._randomize_link_mass and headless,
            visual_model_path=visual_model_path,
        )
        self._configure_model(mjm, cfg, device)
        self._run_task_callbacks(mjm, task)
        self._initialize_link_properties(mjm, num_envs, device)

        viewer_cfg = getattr(cfg, "viewer", None)
        self._show_ui = bool(getattr(viewer_cfg, "show_ui", False))

        # Allocate PyTorch state tensors
        self._dof_state_t = torch.zeros(num_envs, self._num_dof, 2, device=device)
        self._dof_pos_view = self._dof_state_t[..., 0]
        self._dof_vel_view = self._dof_state_t[..., 1]
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (scalar-last, w=1)
        self._rigid_body_states_t = torch.zeros(
            num_envs, self._num_bodies, 13, device=device
        )
        self._rigid_body_states_t[:, :, 6] = 1.0
        self._contact_forces_t = torch.zeros(
            num_envs, self._num_bodies, 3, device=device
        )
        self._contact_friction_t = torch.full(
            (num_envs,), self._nominal_contact_friction, device=device
        )

        # Mass changes require environment-specific derived model constants.
        # The headless model has already discarded visual assets, so private
        # copies remain compact. Other serial worlds share one model.
        if self._randomize_link_mass:
            self._models = [mjm] + [copy.copy(mjm) for _ in range(num_envs - 1)]
            self._datas = [mujoco.MjData(model) for model in self._models]
        else:
            self._datas = [mujoco.MjData(mjm) for _ in range(num_envs)]
        for i, data in enumerate(self._datas):
            self._activate_domain(i)
            model = self._model_for_env(i)
            mujoco.mj_forward(model, data)
            mujoco.mj_rnePostConstraint(model, data)
            self._copy_env_state_from_mujoco(i)

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        torques_np = torques.cpu().numpy()[:, self._native_to_canonical_dof_np]
        off = self._qvel_offset
        for i, d in enumerate(self._datas):
            self._activate_domain(i)
            d.qfrc_applied[off:] = torques_np[i]
            mujoco.mj_step(self._model_for_env(i), d)
        self._sync_state_from_mujoco()
        # Count simulated duration once per batch, independent of world count.
        # A separate clock survives episode resets of individual MjData objects.
        self._render_pending_time += self._mjm.opt.timestep

    def _sync_state_from_mujoco(self) -> None:
        for i, d in enumerate(self._datas):
            self._activate_domain(i)
            # cfrc_ext is only populated with constraint/contact forces by
            # mj_rnePostConstraint; mj_step alone leaves it at zero.
            model = self._model_for_env(i)
            mujoco.mj_rnePostConstraint(model, d)
            # mj_step integrates qpos/qvel after computing body kinematics.
            # Refresh only the derived pose/velocity fields so every public
            # state tensor describes the completed step. A second mj_forward
            # would also recompute collision, constraints, and acceleration.
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            mujoco.mj_comVel(model, d)
            self._copy_env_state_from_mujoco(i)

    def _copy_env_state_from_mujoco(self, env_id: int) -> None:
        """Copy already-refreshed native state into one environment's live views.

        Reset's forward pass has current body kinematics; step refreshes those
        separately after integration. Both paths populate contact forces with
        rnePostConstraint before copying. Keeping this copy local lets partial
        resets preserve every unselected native state and public cache.
        """
        d = self._datas[env_id]
        model = self._model_for_env(env_id)
        dof_order = self._canonical_to_native_dof_np
        body_order = self._canonical_to_native_body_np
        self._dof_pos_view[env_id] = torch.from_numpy(
            d.qpos[self._qpos_offset :][dof_order].copy()
        )
        self._dof_vel_view[env_id] = torch.from_numpy(
            d.qvel[self._qvel_offset :][dof_order].copy()
        )
        self._contact_forces_t[env_id] = torch.from_numpy(
            d.cfrc_ext[body_order, 3:6].copy()
        )
        rbs = self._rigid_body_states_t[env_id]
        rbs[:, 0:3] = torch.from_numpy(d.xpos[body_order].copy())
        mj_quat = torch.from_numpy(d.xquat[body_order].copy())
        rbs[:, 3:7] = mj_quat[:, WXYZ_TO_XYZW]
        angular_velocity = d.cvel[body_order, 0:3]
        root_com = d.subtree_com[model.body_rootid[body_order]]
        body_offset = d.xpos[body_order] - root_com
        linear_velocity = d.cvel[body_order, 3:6] - np.cross(
            body_offset, angular_velocity
        )
        rbs[:, 7:10] = torch.from_numpy(linear_velocity.copy())
        rbs[:, 10:13] = torch.from_numpy(angular_velocity.copy())
        if self._has_free_joint:
            rs = self._root_states_t[env_id]
            rs[:3] = torch.from_numpy(d.qpos[:3].copy())
            mj_quat = torch.from_numpy(d.qpos[3:7].copy())
            rs[3:7] = mj_quat[WXYZ_TO_XYZW]
            rs[7:10] = torch.from_numpy(d.qvel[:3].copy())
            # Free-joint translation is world-frame, but angular qvel is
            # body-local. Public root velocities are both world-frame.
            rs[10:13] = quat_apply(
                mj_quat[WXYZ_TO_XYZW], torch.from_numpy(d.qvel[3:6].copy())
            )

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_state(self, reset_mask: torch.Tensor) -> None:
        qoff = self._qpos_offset
        voff = self._qvel_offset
        for i in reset_mask.nonzero(as_tuple=False).flatten().tolist():
            self._activate_domain(i)
            model = self._model_for_env(i)
            canonical_pos = self._clamp_dof_positions(self._dof_pos_view[i])
            self._dof_pos_view[i].copy_(canonical_pos)
            self._datas[i].qpos[qoff:] = canonical_pos.cpu().numpy()[
                self._native_to_canonical_dof_np
            ]
            self._datas[i].qvel[voff:] = (
                self._dof_vel_view[i].cpu().numpy()[self._native_to_canonical_dof_np]
            )
            if self._has_free_joint:
                rs = self._root_states_t[i].cpu()
                self._datas[i].qpos[:3] = rs[:3].numpy()
                self._datas[i].qpos[3:7] = rs[3:7][XYZW_TO_WXYZ].numpy()
                self._datas[i].qvel[:3] = rs[7:10].numpy()
                # Use the requested orientation, which may change in this reset.
                self._datas[i].qvel[3:6] = quat_rotate_inverse(
                    rs[3:7], rs[10:13]
                ).numpy()
            mujoco.mj_forward(model, self._datas[i])
            mujoco.mj_rnePostConstraint(model, self._datas[i])
            self._copy_env_state_from_mujoco(i)

    def set_all_root_states(self) -> None:
        if not self._has_free_joint:
            return
        for i in range(self._num_envs):
            self._activate_domain(i)
            model = self._model_for_env(i)
            rs = self._root_states_t[i].cpu()
            self._datas[i].qpos[:3] = rs[:3].numpy()
            self._datas[i].qpos[3:7] = rs[3:7][XYZW_TO_WXYZ].numpy()
            self._datas[i].qvel[:3] = rs[7:10].numpy()
            self._datas[i].qvel[3:6] = quat_rotate_inverse(rs[3:7], rs[10:13]).numpy()
            mujoco.mj_forward(model, self._datas[i])
            mujoco.mj_rnePostConstraint(model, self._datas[i])
            self._copy_env_state_from_mujoco(i)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, sync_frame_time: bool = True) -> None:
        import platform
        import mujoco.viewer

        self._activate_domain(0)
        if self._viewer is None:
            if platform.system() == "Darwin":
                import mujoco.viewer as _mjv

                if _mjv._MJPYTHON is None:
                    raise RuntimeError(
                        "MuJoCo passive viewer on macOS requires mjpython.\n"
                        "Run with: .venv/bin/mjpython scripts/train.py ...\n"
                        "Or use --headless to disable the viewer."
                    )

            # Indirect through a wrapper so callers can set _viewer_key_callback
            # after the viewer is up (LeggedRobot.__init__ calls reset()→step()
            # →_render() during construction, before any user interface installs).
            def _key_dispatch(keycode, _self=self):
                if _self._viewer_key_callback is not None:
                    _self._viewer_key_callback(keycode)

            # Side panels are hidden by default: their shortcuts bind most
            # letters to visualisation toggles, which fire alongside keyboard
            # teleop (see gym/utils/interfaces/teleop_bindings.py).
            # cfg.viewer.show_ui / play.py --viewer_ui restores them.
            self._viewer = mujoco.viewer.launch_passive(
                self._mjm,
                self._datas[0],
                key_callback=_key_dispatch,
                show_left_ui=self._show_ui,
                show_right_ui=self._show_ui,
            )
        if self._viewer.is_running():
            now = time.monotonic()
            if sync_frame_time and self._render_deadline is not None:
                # Include every physics substep since the previous render.
                # Rebase after slow frames rather than racing to catch up.
                self._render_deadline = max(
                    self._render_deadline + self._render_pending_time, now
                )
                remaining = self._render_deadline - now
                if remaining > 0:
                    time.sleep(remaining)
            else:
                self._render_deadline = now if sync_frame_time else None
            self._render_pending_time = 0.0
            if self._viewer_overlay_fn is not None:
                self._viewer_overlay_fn(self._viewer)
            self._viewer.sync()
        else:
            self._render_deadline = None
            self._render_pending_time = 0.0
