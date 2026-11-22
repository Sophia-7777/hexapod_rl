"""
MuJoCo simulation entry point with keyboard control.

Keys:
    1       squat
    2       stand
    W / S   forward / backward
    A / D   left / right
    Q / E   yaw left / yaw right
    SPACE   stop and stand
    ESC     quit
"""

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from pynput import keyboard

from envs.SopEngine import HexapodEngine, RobotState


class KeyboardController:
    def __init__(self, engine):
        self.engine = engine
        self.pressed = set()
        self.lock = threading.Lock()
        self.listener = None

    def _movement_command(self):
        vx = float("w" in self.pressed) - float("s" in self.pressed)
        vy = float("a" in self.pressed) - float("d" in self.pressed)
        wz = float("q" in self.pressed) - float("e" in self.pressed)

        # Normalize translation so diagonal movement is not faster.
        norm = (vx * vx + vy * vy) ** 0.5
        if norm > 1.0:
            vx /= norm
            vy /= norm
        return vx, vy, wz

    def on_press(self, key):
        try:
            value = key.char.lower()
        except AttributeError:
            value = None

        if value in ("w", "a", "s", "d", "q", "e"):
            with self.lock:
                self.pressed.add(value)
                command = self._movement_command()
            self.engine.set_command(*command)
            return

        if value == "1":
            self.engine.request_state(RobotState.SQUAT)
        elif value == "2":
            self.engine.request_state(RobotState.STAND)

        if key == keyboard.Key.space:
            with self.lock:
                self.pressed.clear()
            self.engine.stop()
            self.engine.request_state(RobotState.STAND)

        if key == keyboard.Key.esc:
            return False

    def on_release(self, key):
        try:
            value = key.char.lower()
        except AttributeError:
            value = None

        if value in ("w", "a", "s", "d", "q", "e"):
            with self.lock:
                self.pressed.discard(value)
                command = self._movement_command()
            self.engine.set_command(*command)

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default=str(
            Path.home()
            / "RESEARCH"
            / "hexapod"
            / "stable-baselines3_ppo"
            / "hexapod_rl_final1"
            / "envs"
            / "assets"
            / "hexapod_mujoco2"
            / "plane_scene.xml"
        ),
        help="MuJoCo XML model path",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    engine = HexapodEngine(model, data)
    keyboard_controller = KeyboardController(engine)
    keyboard_controller.start()

    physics_dt = model.opt.timestep
    control_dt = engine.control_period
    next_control_time = time.perf_counter()

    print(
        "\n"
        "========== HEXAPOD CONTROL ==========\n"
        "  1       : Squat\n"
        "  2       : Stand\n"
        "  W / S   : Forward / Backward\n"
        "  A / D   : Left / Right\n"
        "  Q / E   : Turn Left / Turn Right\n"
        "  SPACE   : Stop and Stand\n"
        "  ESC     : Exit\n"
        "=====================================\n"
    )

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                now = time.perf_counter()

                if now >= next_control_time:
                    engine.update(control_dt)
                    next_control_time += control_dt

                    # Avoid accumulating a large time backlog after a pause.
                    if now - next_control_time > 0.2:
                        next_control_time = now + control_dt

                mujoco.mj_step(model, data)
                viewer.sync()

                # Pace simulation approximately at real-time speed.
                time.sleep(max(0.0, physics_dt * 0.5))

    except KeyboardInterrupt:
        pass
    finally:
        keyboard_controller.stop()


if __name__ == "__main__":
    main()