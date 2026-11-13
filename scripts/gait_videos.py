"""Short videos (animated GIFs) of example gaits, one per gait.

Each video runs a batch of robots exactly as grid_eval.py does (random
initial phases, settle, then measure), classifies every robot the same way,
and renders the robot most typical of the batch's most common label (as in
gait_examples.py) over the last seconds of the run, in slow motion. A strip
at the bottom lights each foot while it is on the ground.

Usage:
    uv run --frozen scripts/gait_videos.py               # front-locked gaits
    uv run --frozen scripts/gait_videos.py --slowdown 3 --seconds 2
"""

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

from gym.utils.helpers import GYM_ROOT_DIR

# (task, run, speed [m/s], omega [Hz], file name)
EXAMPLES = [
    ("go2osc_front_locked_omega", "Nov10_05-47-12_", 1.0, 3.0, "hop"),
    ("go2osc_front_locked_omega", "Nov10_07-55-19_", 1.5, 3.0, "rear_first_hop"),
    ("go2osc_front_locked_omega_gait2", "Nov10_10-00-19_", 1.5, 3.0, "diagonal_pair_then_RL"),
    ("go2osc_front_locked_omega", "Nov10_07-55-19_", 1.0, 1.5, "irregular"),
]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Videos of example gaits")
    parser.add_argument("--robots", type=int, default=32, help="robots per example")
    parser.add_argument("--seconds", type=float, default=2.0, help="simulated time shown [s]")
    parser.add_argument("--slowdown", type=float, default=2.0, help="playback slow-down")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--jobs", type=int, default=4, help="parallel simulations")
    parser.add_argument(
        "--view",
        type=float,
        default=135.0,
        help="camera azimuth relative to the robot's heading [deg]: 0 looks "
        "along its heading from behind, 90 from its right side, 135 from "
        "front-right (shows the locked FR leg)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(GYM_ROOT_DIR) / "logs" / "sweep_results" / "gait_videos"),
    )
    # Internal: render one example (JSON).
    parser.add_argument("--one", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def render(example, args):
    import torch
    from PIL import Image, ImageDraw

    import grid_eval
    import play
    from gait_examples import typical_robot
    from gym.utils.interfaces.VideoRecorder import VideoRecorder

    task, run, speed, omega, name = example
    eval_args = grid_eval.get_args(
        ["--task", task, "--load_run", run, "--robots", str(args.robots)]
    )
    # --record (with --headless) keeps the visual meshes the renderer needs.
    play_argv = ["--task", task, "--load_run", run, "--headless", "--no-keyboard",
                 "--record", "--num_envs", str(args.robots), "--seed", "0"]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
        # Buffer every robot's configuration over the last `seconds`.
        datas = env._backend._datas
        poses = deque(maxlen=int(round(args.seconds / env.dt)))
        step = env.step

        def recording_step():
            step()
            poses.append(np.stack([d.qpos.copy() for d in datas]))

        env.step = recording_step
        log = grid_eval.rollout(env, runner, speed, eval_args, omega)
        env.step = step

    pick = typical_robot(env, log, eval_args)
    robot, legs, names = pick["robot"], pick["legs"], pick["names"]
    contact = log["contact_all"][robot][-len(poses):][:, legs]
    vx = log["vx"][robot]

    # Frames: every `stride`-th control step, played at `fps` -> slowdown.
    stride = max(1, round(1.0 / (args.fps * env.dt * args.slowdown)))
    frames = list(range(0, len(poses), stride))
    fps = 1.0 / (stride * env.dt * args.slowdown)
    caption = (
        f"{grid_eval.robot_label(env.cfg.leg_lock)}: {pick['label']} "
        f"({pick['share']:.0%} of robots)\n"
        f"{task} {run}  command {speed:g} m/s (measured {vx:.2f}), "
        f"oscillator {omega:g} Hz, {args.slowdown:g}x slow motion"
    )

    import mujoco

    def heading(qpos):
        w, x, y, z = qpos[3:7]
        return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    # Camera turns with the robot (heading smoothed over ~0.5 s) so every
    # video shows the same side of it, whatever direction it ran.
    yaw = np.unwrap(np.radians([heading(poses[k][robot]) for k in frames]))
    n = max(1, int(round(0.5 * fps)))
    yaw = np.degrees(np.convolve(np.pad(yaw, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), "valid"))

    class CaptionedRecorder(VideoRecorder):
        def _frames(self, renderer):
            data = mujoco.MjData(self._model)
            for qpos, k, heading_deg in zip(self._qpos, frames, yaw):
                data.qpos[:] = qpos
                mujoco.mj_forward(self._model, data)
                self._camera.azimuth = heading_deg + args.view
                renderer.update_scene(data, self._camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 34), fill=(255, 255, 255))
                draw.text((6, 3), caption, fill=(26, 26, 25))
                # Foot-contact strip: filled box while the foot is loaded.
                for i, leg in enumerate(names):
                    x = 10 + 70 * i
                    y = image.height - 30
                    on = contact[k, i]
                    draw.rectangle((x, y, x + 60, y + 22),
                                   fill=(42, 120, 214) if on else (255, 255, 255),
                                   outline=(26, 26, 25))
                    draw.text((x + 20, y + 6), leg, fill=(255, 255, 255) if on else (26, 26, 25))
                yield image

    out = Path(args.out_dir) / f"{name}.gif"
    recorder = CaptionedRecorder(env, out, seconds=len(frames) / fps, fps=fps)
    recorder._model = env._backend._model_for_env(robot)
    recorder._qpos = deque(poses[k][robot] for k in frames)
    recorder.fps = fps
    recorder.save()
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"task": task, "run": run, "speed": speed, "omega": omega,
                   "robot": int(robot), "label": pick["label"],
                   "share": float(pick["share"]), "timing": pick["timing"].tolist(),
                   "model": f"{Path(load_path).parent.name}/{Path(load_path).stem}"}, f)


def main():
    args = get_args()
    if args.one is not None:
        render(tuple(json.loads(args.one)), args)
        return
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, 16 // args.jobs)))
    running = []
    for example in EXAMPLES:
        cmd = [sys.executable, __file__, "--one", json.dumps(example),
               "--robots", str(args.robots), "--seconds", str(args.seconds),
               "--slowdown", str(args.slowdown), "--fps", str(args.fps),
               "--out_dir", args.out_dir]
        running.append(subprocess.Popen(cmd, env=env))
        if len(running) >= args.jobs:
            running.pop(0).wait()
    for p in running:
        p.wait()


if __name__ == "__main__":
    main()
