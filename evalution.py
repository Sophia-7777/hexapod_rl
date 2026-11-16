import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.hexapod_env import HexapodEnv


# -----------------------------------------------------
# 1. 单环境创建
# -----------------------------------------------------
def make_env(xml_path):
    def _init():
        env = HexapodEnv(xml_path=xml_path, render_mode="human")
        return env
    return _init


# -----------------------------------------------------
# 2. 评估函数（不再包含 parse_step_output）
# -----------------------------------------------------
def evaluate(model_path="ppo_hexapod_final.zip",
             vecnorm_path="vec_normalize.pkl",
             xml_path="hexapod.xml",
             episodes=5):
    # == 创建 VecEnv ==
    eval_env = DummyVecEnv([make_env(xml_path)])

    # == 加载 VecNormalize ==
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False      # 必须关掉训练模式
        eval_env.norm_reward = False   # 不归一化奖励
    else:
        print("WARNING: VecNormalize 未找到，将直接评估原始环境！")

    # == 加载模型 ==
    model = PPO.load(model_path, env=eval_env, device="cpu")

    # == 开始评估 ==
    for ep in range(episodes):
        
        obs = eval_env.reset()
        done = False
        ep_reward = 0

        while not done:
            # 模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # VecEnv.step() 始终返回 4 个值
            obs, reward, done_array, info = eval_env.step(action)

            done = bool(done_array[0])  # VecEnv 的 done 是数组
            ep_reward += reward[0]
        print("1")
        print(f"[Episode {ep + 1}] Reward = {ep_reward:.3f}")

    print("Evaluation finished!")


# -----------------------------------------------------
# 3. 主入口
# -----------------------------------------------------
if __name__ == "__main__":
    evaluate()

