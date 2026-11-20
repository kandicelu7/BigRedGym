"""Records the most recent part of a MuJoCo playback to an animated GIF / WebP.

During playback only env 0's joint configuration (qpos) is buffered, which is
nearly free, so playback speed is unaffected. save() then replays the last
`seconds` of configurations offscreen with a camera that follows the robot's
base (like CameraTracker) and writes the frames. Uses Pillow only, so no video
codec needs to be installed; the output format follows the file extension
(.gif or .webp).
"""

from collections import deque
from pathlib import Path

import mujoco
from PIL import Image


class VideoRecorder:
    def __init__(
        self,
        env,
        path,
        seconds=20.0,
        fps=25,
        width=640,
        height=480,
        body="base",
        distance=1.5,
        azimuth=150,
        elevation=-20,
    ):
        backend = env._backend
        if not hasattr(backend, "_datas"):
            raise RuntimeError(
                "video recording needs the MuJoCo CPU backend; use --device cpu"
            )
        self.path = Path(path)
        self.size = (height, width)
        self._model = backend._model_for_env(0)
        self._data = backend._datas[0]

        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, body
        )
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation

        # Keep every `stride`-th control step; fps is what that works out to.
        self._stride = max(1, round(1.0 / (fps * env.dt)))
        self.fps = 1.0 / (self._stride * env.dt)
        self._qpos = deque(maxlen=max(1, int(seconds * self.fps)))
        self._step = 0

    def capture(self):
        """Call once per control step, after env.step()."""
        if self._step % self._stride == 0:
            self._qpos.append(self._data.qpos.copy())
        self._step += 1

    def _frames(self, renderer):
        data = mujoco.MjData(self._model)
        for qpos in self._qpos:
            data.qpos[:] = qpos
            mujoco.mj_forward(self._model, data)
            renderer.update_scene(data, self._camera)
            yield Image.fromarray(renderer.render())

    def save(self):
        if not self._qpos:
            return None
        seconds = len(self._qpos) / self.fps
        print(f"Rendering last {seconds:.1f} s of playback ({len(self._qpos)} frames)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Closed here, while GLFW is still initialized: the viewer terminates
        # GLFW at exit, after which renderer cleanup logs an error.
        with mujoco.Renderer(self._model, *self.size) as renderer:
            frames = self._frames(renderer)
            first = next(frames)
            first.save(
                self.path,
                save_all=True,
                append_images=frames,
                duration=round(1000 / self.fps),
                loop=0,
            )
        print(f"Saved playback video to {self.path}")
        return self.path
