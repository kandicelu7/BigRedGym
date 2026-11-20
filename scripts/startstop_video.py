"""Video (animated GIF) of policies through the start/stop test.

Runs the command sequence of startstop_eval.py (stand, ramp up, turn, stop,
stand, restart) for a batch of robots per policy, renders the most typical
robot of each (the one whose stepping-while-stopped is closest to the
batch median), and puts the policies side by side. Each panel shows the
current command, the measured forward speed and which feet are loaded.

Usage:
    uv run --frozen scripts/startstop_video.py \
        --policy go2osc_wide_even2.5_deploy Nov16_09-10-17_seed4 "deploy seed 4" \
        --policy go2osc_wide_baseline <run> "baseline"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gym.utils.helpers import GYM_ROOT_DIR

OUT = Path(GYM_ROOT_DIR) / "results" / "study" / "startstop" / "videos"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--policy",
        nargs=3,
        action="append",
        metavar=("TASK", "RUN", "TITLE"),
        help="a policy to show (repeat for side-by-side panels)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="test speed [m/s]")
    parser.add_argument(
        "--omega", type=float, default=3.0, help="moving frequency [Hz]"
    )
    parser.add_argument("--robots", type=int, default=16)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument(
        "--view",
        type=float,
        default=135.0,
        help="camera azimuth relative to the heading [deg]",
    )
    parser.add_argument(
        "--ramp_only",
        action="store_true",
        help="stand, ramp up, hold, ramp down and stand (no turn, no restart)",
    )
    parser.add_argument(
        "--slowdown", type=float, default=1.0, help="playback slow-down (2: half speed)"
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--one", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def phase_text(t, speed):
    import startstop_eval as se

    if t < se.T_START:
        return "command: STAND"
    if t < se.T_TURN:
        return f"command: GO {speed:g} m/s"
    if t < se.T_TURN_END:
        return f"command: GO + TURN {se.YAW_RATE:g} rad/s"
    if t < se.T_STOP:
        return f"command: GO {speed:g} m/s"
    if t < se.T_RESTART:
        return "command: STOP"
    return f"command: GO {speed:g} m/s"


def render_one(task, run, title, args, out):
    import mujoco
    import torch
    from PIL import Image, ImageDraw

    import play
    import startstop_eval as se

    import gym.envs.go2.go2osc_config  # noqa: F401

    play_argv = [
        "--task",
        task,
        "--load_run",
        run,
        "--headless",
        "--no-keyboard",
        "--record",
        "--num_envs",
        str(args.robots),
        "--seed",
        "0",
        "--device",
        "cpu",
    ]
    with torch.no_grad():
        env, runner, load_path = play.setup(play.get_play_args(play_argv))
        datas = env._backend._datas
        active = env.active_legs.cpu().numpy()
        poses, contact, vx = [], [], []
        step = env.step

        def recording_step():
            step()
            poses.append(np.stack([d.qpos.copy() for d in datas]))
            contact.append(env.grf.cpu().numpy()[:, active] > se.TOUCH)
            vx.append(env.base_lin_vel[:, 0].cpu().numpy().copy())  # not a view

        env.step = recording_step
        if args.ramp_only:
            # Drop the turn and the restart: stand 0-3 s, ramp up and hold
            # until 12 s, ramp down and stand to 17 s.
            se.T_TURN_END = se.T_TURN
            se.T_RESTART = se.T_END = se.T_STOP + 5.0
        res = se.rollout(env, runner, args.speed, args.omega)
        env.step = step
    poses, contact, vx = np.stack(poses), np.stack(contact), np.stack(vx)
    # The typical robot: stepping while stopped closest to the batch median.
    steps = np.where(res["fell"], np.nan, res["steps"])
    robot = int(np.nanargmin(np.abs(steps - np.nanmedian(steps))))
    stopped = np.isfinite(res["stop_time"])[~res["fell"]].mean()
    legs = [n for n, a in zip(["FL", "FR", "RL", "RR"], active) if a]

    dt = env.dt
    stride = max(1, round(1.0 / (args.fps * dt)))
    frames = list(range(0, len(poses), stride))

    def heading(q):
        w, x, y, z = q[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    yaw = np.unwrap([heading(poses[k][robot]) for k in frames])
    n = max(1, int(round(0.5 * args.fps)))
    yaw = np.degrees(
        np.convolve(
            np.pad(yaw, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), "valid"
        )
    )

    model = env._backend._model_for_env(robot)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam.distance, cam.elevation = 1.6, -20.0
    data = mujoco.MjData(model)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    smooth = np.convolve(vx[:, robot], np.ones(20) / 20, "same")
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for j, (k, az) in enumerate(zip(frames, yaw)):
            data.qpos[:] = poses[k][robot]
            mujoco.mj_forward(model, data)
            cam.azimuth = az + args.view
            renderer.update_scene(data, cam)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            t = (k + 1) * dt
            cx, _ = se.command_profile(t, args.speed)
            d.rectangle((0, 0, img.width, 46), fill=(255, 255, 255))
            d.text(
                (6, 3), f"{title}  ({stopped:.0%} of robots stop)" + (f"  {1 / args.slowdown:g}x speed" if args.slowdown != 1 else ""), fill=(26, 26, 25)
            )
            d.text(
                (6, 17),
                phase_text(t, args.speed),
                fill=(200, 40, 40)
                if "STOP" in phase_text(t, args.speed)
                or "STAND" in phase_text(t, args.speed)
                else (26, 26, 25),
            )
            d.text(
                (6, 31),
                f"t {t:4.1f} s   cmd {cx:.2f} m/s   actual {smooth[k]:+.2f} m/s",
                fill=(26, 26, 25),
            )
            for i, leg in enumerate(legs):
                x, y = 8 + 52 * i, img.height - 26
                on = contact[k, robot, i]
                d.rectangle(
                    (x, y, x + 46, y + 20),
                    fill=(42, 120, 214) if on else (255, 255, 255),
                    outline=(26, 26, 25),
                )
                d.text(
                    (x + 16, y + 5), leg, fill=(255, 255, 255) if on else (26, 26, 25)
                )
            img.save(out_dir / f"{j:04d}.png")
    meta = {
        "task": task,
        "run": run,
        "robot": robot,
        "stopped": float(stopped),
        "fps": len(frames) / (len(poses) * dt),
        "model": str(load_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"{title}: robot {robot}, {stopped:.0%} of robots stop, {len(frames)} frames")


def main():
    args = get_args()
    if args.one is not None:
        task, run, title, out = json.loads(args.one)
        render_one(task, run, title, args, out)
        return
    from PIL import Image

    tmp = OUT / "_frames"
    procs = []
    for i, (task, run, title) in enumerate(args.policy):
        cmd = [
            sys.executable,
            __file__,
            "--one",
            json.dumps([task, run, title, str(tmp / str(i))]),
        ]
        cmd += sys.argv[1:]
        procs.append(subprocess.Popen(cmd))
    if any(p.wait() for p in procs):
        raise SystemExit("a panel failed")
    panels = [sorted((tmp / str(i)).glob("*.png")) for i in range(len(args.policy))]
    fps = json.loads((tmp / "0" / "meta.json").read_text())["fps"]
    count = min(len(p) for p in panels)

    def canvas_at(j):
        ims = [Image.open(p[j]).convert("RGB") for p in panels]
        canvas = Image.new(
            "RGB",
            (sum(im.width for im in ims) + 4 * (len(ims) - 1), ims[0].height),
            (255, 255, 255),
        )
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0))
            x += im.width + 4
        return canvas

    # One palette for every frame (from a few spread-out frames): per-frame
    # palettes break the GIF writer's frame differencing (stale regions).
    samples = [canvas_at(j) for j in np.linspace(0, count - 1, 12).astype(int)]
    strip = Image.new("RGB", (samples[0].width, samples[0].height * len(samples)))
    for i, s in enumerate(samples):
        strip.paste(s, (0, i * s.height))
    palette = strip.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    frames = [
        canvas_at(j).quantize(palette=palette, dither=Image.Dither.NONE)
        for j in range(count)
    ]
    out = Path(args.out or OUT / "startstop.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=round(1000 * args.slowdown / fps), loop=0
    )
    import shutil

    shutil.rmtree(tmp)
    print(f"Saved {out} ({count} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
