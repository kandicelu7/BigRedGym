"""Keeps the MuJoCo viewer camera following the robot of env 0.

Switches the passive viewer to MuJoCo's tracking camera on the robot's base
body: the camera translates with the robot but keeps its own orientation, so
mouse rotate / zoom still work while it follows.

Chains onto the backend's per-frame overlay hook, so install it after any
other overlay (e.g. CommandVisualizer).
"""

import mujoco


class CameraTracker:
    def __init__(self, env, body="base", distance=1.5, azimuth=150, elevation=-20):
        self.body = body
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self._configured = False
        # Only the CPU MuJoCo backend has an interactive viewer (see
        # CommandVisualizer); elsewhere this is a no-op.
        backend = env._backend
        if not hasattr(backend, "_viewer_overlay_fn"):
            return
        self._prev_overlay = backend._viewer_overlay_fn
        backend._viewer_overlay_fn = self._update

    def _update(self, viewer):
        if not self._configured:
            body_id = mujoco.mj_name2id(viewer.m, mujoco.mjtObj.mjOBJ_BODY, self.body)
            if body_id < 0:
                raise ValueError(f"no body named {self.body!r} to track")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = body_id
            viewer.cam.distance = self.distance
            viewer.cam.azimuth = self.azimuth
            viewer.cam.elevation = self.elevation
            self._configured = True
        if self._prev_overlay is not None:
            self._prev_overlay(viewer)
