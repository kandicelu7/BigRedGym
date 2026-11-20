"""Drive an oscillator policy with the keyboard at a fixed oscillator frequency.

Like play.py (MuJoCo viewer, keyboard teleop, camera following robot 0), but
the oscillator frequency is held at --omega, and each key press changes the
held speed or turn command by a fixed step; it stays until the next press (for
policies trained with a command ramp, commands.max_accel, the speed command
ramps to it at the trained acceleration). Each press is printed at once, and
a status line shows speed, turn, foot contacts (# on the ground, . in the air)
and the gait, classified as in the grid evaluation (grid_eval.robot_gait) from
the last --gait_window seconds of foot contacts. Forward commands below
osc.stop_threshold (0.5 m/s) mean stop: the policy stands.

Keys (MuJoCo viewer window focused):
    I / K   speed +/- --speed_step (0.5 m/s)   N / M   turn +/- --yaw_step (0.2 rad/s)
    R       reset                               Esc     quit

Usage:
    uv run --frozen scripts/teleop_osc.py --task go2osc_wide_deploy_robust \
        --load_run Nov16_23-40-18_seed2 --omega 3.75
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument(
        "--omega", type=float, default=3.75, help="oscillator frequency [Hz]"
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--speed_step", type=float, default=0.5, help="[m/s] per press")
    parser.add_argument("--yaw_step", type=float, default=0.2, help="[rad/s] per press")
    parser.add_argument("--max_speed", type=float, default=2.5, help="[m/s]")
    parser.add_argument("--max_yaw", type=float, default=1.0, help="[rad/s]")
    parser.add_argument(
        "--gait_window", type=float, default=2.5, help="[s] of contacts classified"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # print live, even through a pipe

    play_args = play.get_play_args(
        [
            "--task",
            args.task,
            "--load_run",
            args.load_run,
            "--num_envs",
            str(args.num_envs),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--no-record",
        ]
    )
    with torch.no_grad():
        env, runner, load_path = play.setup(play_args)
        from gym.utils.interfaces.CameraTracker import CameraTracker
        from gym.utils.interfaces.CommandVisualizer import CommandVisualizer
        from gym.utils.interfaces.MujocoKeyboardInterface import (
            KEYCODE_TO_ACTION,
            MujocoKeyboardInterface,
        )

        keyboard = MujocoKeyboardInterface(env)
        CommandVisualizer(env)
        CameraTracker(env)
        ramped = getattr(env.cfg.commands, "max_accel", None) is not None
        stop_below = getattr(env.cfg.osc, "stop_threshold", 0.0)
        # The held commands: each key press changes them by a fixed step and
        # they stay until the next press.
        held = {"speed": 0.0, "yaw": 0.0}

        def show(event):
            stop = "  (stop: stands)" if held["speed"] < stop_below else ""
            print(
                f"\n[{env.common_step_counter * env.dt:6.1f} s] {event}: speed "
                f"command {held['speed']:+.2f} m/s, turn command "
                f"{held['yaw']:+.2f} rad/s{stop}",
                flush=True,
            )

        def on_key(keycode):
            # Replaces play.py's handler, which adds to env.commands directly
            # (the trained command ramp then pulls that back down).
            action = KEYCODE_TO_ACTION.get(keycode)
            if action == "forward":
                held["speed"] = min(held["speed"] + args.speed_step, args.max_speed)
            elif action == "back":
                held["speed"] = max(held["speed"] - args.speed_step, 0.0)
            elif action == "yaw_left":
                held["yaw"] = min(held["yaw"] + args.yaw_step, args.max_yaw)
            elif action == "yaw_right":
                held["yaw"] = max(held["yaw"] - args.yaw_step, -args.max_yaw)
            elif action == "reset":
                keyboard.commands.apply("reset")
            else:
                return
            held["speed"], held["yaw"] = round(held["speed"], 3), round(held["yaw"], 3)
            show(f"key {action}")

        env._backend._viewer_key_callback = on_key

        # Live gait label: grid_eval's classifier (as in the heat maps) on the
        # last --gait_window seconds of foot contacts of robot 0.
        import grid_eval
        from collections import deque

        import numpy as np

        gargs = grid_eval.get_args([])
        legs = [k for k, a in enumerate(env.active_legs.tolist()) if a]
        feet = env.robot_layout.body_groups["feet"]
        leg_names = [feet[k].removesuffix("_foot") for k in legs]
        ref, others, gaits = grid_eval.gait_reference(legs, leg_names)
        history = deque(maxlen=int(round(args.gait_window / env.dt)))
        gait = ["(collecting)"]

        def classify_gait():
            contact = np.array(history)
            if len(contact) < history.maxlen:
                return "(collecting)"
            # The classifier calls a robot that never steps 'irregular' (the
            # grid evaluation never stands still): at most one touchdown per
            # foot in the window counts as standing.
            min_air = int(round(0.03 / env.dt))
            steps = [
                len(grid_eval.footfall.touchdowns(contact[:, k], min_air)) for k in legs
            ]
            if max(steps) <= 1:
                return "standing (not stepping)"
            try:
                return grid_eval.robot_gait(
                    contact, ref, others, legs, gaits, env.dt, gargs
                )[3]
            except Exception:  # too few strides in the window, etc.
                return "irregular"

        env.commands[:] = 0.0  # start standing
        step = env.step

        def teleop_step():
            env.episode_omega[:] = args.omega
            moving = held["speed"] >= stop_below
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = held["yaw"] if moving else 0.0  # trained: no turning
            if ramped:  # the command ramps to the held speed at the trained rate
                env.command_target[:] = held["speed"]
            else:
                env.commands[:, 0] = held["speed"]
            step()
            on = (env.grf[0] > gargs.contact_threshold).cpu().numpy()
            history.append(on)
            if env.episode_length_buf[0] <= 1:  # fell and reset: start over
                history.clear()
            if env.common_step_counter % 25 == 0:
                gait[0] = classify_gait()
                feet_now = " ".join(
                    f"{n}{'#' if on[k] else '.'}" for n, k in zip(leg_names, legs)
                )
                print(
                    f"\rspeed: cmd {env.commands[0, 0]:+.2f} (held {held['speed']:+.2f})"
                    f" actual {env.base_lin_vel[0, 0]:+.2f} m/s | turn: cmd "
                    f"{env.commands[0, 2]:+.2f} actual {env.base_ang_vel[0, 2]:+.2f}"
                    f" rad/s | feet {feet_now} | gait: {gait[0]:<28}",
                    end="",
                    flush=True,
                )

        env.step = teleop_step
        print(
            f"Loaded {load_path}\nOscillator held at {args.omega:g} Hz. Focus the "
            f"viewer window: I/K speed +-{args.speed_step:g} m/s, N/M turn "
            f"+-{args.yaw_step:g} rad/s, R reset, Esc quit."
        )
        play.play(env, runner)


if __name__ == "__main__":
    main()
