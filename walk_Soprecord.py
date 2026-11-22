"""
MuJoCo hexapod forward walking simulation with MP4 recording.

Features:
    - Automatic forward walking
    - Low-angle side camera
    - Camera follows the robot
    - Record simulation as MP4 video
"""

import argparse
import time
from pathlib import Path

import mujoco
import imageio.v2 as imageio

from envs.SopEngine import HexapodEngine


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

    parser.add_argument(
        "--output",
        default="hexapod_forward_walking.mp4",
        help="Output video path",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Simulation duration in seconds",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate",
    )

    args = parser.parse_args()

    # ==========================================
    # 1. Load MuJoCo model
    # ==========================================

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    engine = HexapodEngine(model, data)

    physics_dt = model.opt.timestep
    control_dt = engine.control_period

    next_control_time = time.perf_counter()

    # ==========================================
    # 2. Set forward walking command
    # ==========================================

    engine.set_command(1.0, 0.0, 0.0)

    print("\n")
    print("========== HEXAPOD VIDEO RECORDING ==========")
    print("  Mode       : Automatic Forward Walking")
    print("  Velocity   : Forward")
    print("  Duration   :", args.duration, "seconds")
    print("  FPS        :", args.fps)
    print("  Output     :", args.output)
    print("=============================================\n")

    # ==========================================
    # 3. Configure camera
    # ==========================================

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    # Low-angle oblique side view.
    camera.azimuth = 135
    camera.elevation = -12

    # Camera distance.
    camera.distance = 2.0

    # Initial camera target.
    camera.lookat[:] = [0.0, 0.0, 0.18]

    # ==========================================
    # 4. Initialize renderer
    # ==========================================

    # ==========================================
# 4. Initialize renderer
# ==========================================

    width = 1280
    height = 720

    # Increase offscreen framebuffer dimensions.
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    # ==========================================
    # 5. Initialize video writer
    # ==========================================

    output_path = Path(args.output)

    writer = imageio.get_writer(
        str(output_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    # ==========================================
    # 6. Simulation and video recording
    # ==========================================

    total_steps = int(args.duration / physics_dt)

    frame_interval = 1.0 / args.fps
    next_frame_time = 0.0

    simulation_time = 0.0

    try:

        for step in range(total_steps):

            now = time.perf_counter()

            # ----------------------------------
            # Update robot controller
            # ----------------------------------

            if now >= next_control_time:

                engine.update(control_dt)

                next_control_time += control_dt

                # Avoid accumulating a large time backlog.
                if now - next_control_time > 0.2:
                    next_control_time = now + control_dt

            # ----------------------------------
            # Physics simulation
            # ----------------------------------

            mujoco.mj_step(model, data)

            simulation_time += physics_dt

            # ----------------------------------
            # Camera tracking
            # ----------------------------------

            # Assuming the robot's main body is
            # the first body after worldbody.
            robot_position = data.xpos[1].copy()

            camera.lookat[0] = robot_position[0]
            camera.lookat[1] = robot_position[1]
            camera.lookat[2] = 0.18

            # ----------------------------------
            # Render and record video
            # ----------------------------------

            if simulation_time >= next_frame_time:

                renderer.update_scene(
                    data,
                    camera=camera,
                )

                frame = renderer.render()

                writer.append_data(frame)

                next_frame_time += frame_interval

            # ----------------------------------
            # Progress information
            # ----------------------------------

            if step % max(1, int(physics_dt * 100)) == 0:

                progress = simulation_time / args.duration * 100

                print(
                    f"\rRecording: {progress:.1f}% "
                    f"| Simulation time: {simulation_time:.2f}s",
                    end="",
                    flush=True,
                )

        print("\n\nSimulation finished.")

    except KeyboardInterrupt:

        print("\nRecording interrupted.")

    finally:

        writer.close()
        renderer.close()

        print("\n=============================================")
        print("Video saved successfully:")
        print(output_path.resolve())
        print("=============================================\n")


if __name__ == "__main__":
    main()