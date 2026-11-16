import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from envs.hexapod_env import HexapodEnv

# 加载训练好的模型
model = PPO.load("ppo_hexapod_final.zip")

# 创建环境
env = HexapodEnv(xml_path="narrow_scene.xml", render_mode="human")

# 运行评估
obs = env.reset()[0]

# 评估过程
for _ in range(99999):
    action, _states = model.predict(obs,deterministic = True)
    obs, rewards, terminated, truncated, info = env.step(action)

    # 打印调试信息
    # print("Step completed")

    # 如果 done 或 truncated，重置环境
    if terminated or truncated:
        print("Environment reset")
        obs = env.reset()[0]  # 重置环境
        

    # 确保渲染过程在每一步都被调用
    env.render()