import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from envs.hexapod_env import HexapodEnv
import imageio
import torch
import subprocess
import mujoco

os.makedirs("frames", exist_ok=True)

def record(env, model, filename, fps, max_steps=3000):
    frames = []

    obs, _ = env.reset()
    height, width = 480, 640

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "18",
            "hexapod.mp4",
        ],
        stdin=subprocess.PIPE,
    )

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()   # 获取渲染帧（已经按照新的摄像头设置渲染）

        # 保存帧
        imageio.imwrite(f"frames/{step:03d}.png", frame)

        frames.append(frame)
        proc.stdin.write(frame.tobytes())

    print("total frames:", len(frames))
    proc.stdin.close()
    proc.wait()

model = PPO.load("ppo_hexapod_final.zip", device="cpu")

env = HexapodEnv(
    xml_path="narrow_scene.xml",
    render_mode="rgb_array"  # 使用 rgb_array 渲染模式
)

# xml_path = "narrow_scene.xml"
# if not os.path.isabs(xml_path):
#     xml_path = os.path.join(
#         os.path.dirname(__file__),
#         "envs",
#         "assets",
#         "hexapod_mujoco2",
#         xml_path,
#     )
# model = mujoco.MjModel.from_xml_path(xml_path)
# data  = mujoco.MjData(model)
# print("开始录制")
# dt = model.opt.timestep
# fps = int(1.0 / dt)
record(env, model, filename="hexapod.mp4", fps=200)
print("录制完成")

env.close()
