import os
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.hexapod_env import HexapodEnv  # 你的自定义环境
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------
# 1. CPU-friendly PPO Policy
# ---------------------------------------------------------
policy_kwargs = dict(
   net_arch=dict(
        pi=[512, 256, 128],   # 策略网络隐藏层
        vf=[512, 256, 128]    # 价值网络隐藏层
    ),
    activation_fn=torch.nn.ELU,  # 激活函数 ELU
    ortho_init=True,             # 使用正交初始化，通常 PPO 推荐
    log_std_init=0.0             # 初始 log_std，可根据需要调节探索
)

# ---------------------------------------------------------
# 2. 多环境创建函数
# ---------------------------------------------------------
def make_env(rank, xml_path, seed=0):
    def _init():
        env = HexapodEnv(xml_path=xml_path)
        env.seed(seed + rank)
        return env
    return _init


# ---------------------------------------------------------
# 3. 启动训练
# ---------------------------------------------------------
if __name__ == "__main__":
    xml_path = "narrow_scene.xml"

    # CPU-friendly。使用 4 个并行环境
    num_envs = 4

    # ---------------------------------------------------------
    # 创建多环境
    # ---------------------------------------------------------
    envs = SubprocVecEnv([make_env(i, xml_path) for i in range(num_envs)])

    # ---------------------------------------------------------
    # 状态归一化（非常重要）
    # 注意：多足机器人 reward 不建议归一化
    # ---------------------------------------------------------
    # envs = VecNormalize(
    #     envs,
    #     norm_obs=True,        # 对观测值进行归一化
    #     norm_reward=False,    # ★ 关键修改：不归一化 reward
    #     clip_obs=10.0
    # )

    # ---------------------------------------------------------
    # 模型保存回调
    # ---------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,            # 每 100k step 保存一次
        save_path="./checkpoints/",   # 模型保存路径
        name_prefix="ppo_hexapod"     # 文件名前缀
    )

    # ---------------------------------------------------------
    # PPO 模型
    # ---------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=0.001,             
        n_steps=250,                    
        batch_size=250,
        n_epochs=5,                       
        gamma=0.99,
        gae_lambda=0.95,    
        clip_range=0.15,
        ent_coef=0.0015,
        # vf_coef=1.0,
        # max_grad_norm=1.0,
        # target_kl=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./hexapod_tensorboard/",
        device="cpu"
    )


    # ---------------------------------------------------------
    # 自定义 Reward Callback（记录未归一化 reward）
    # ---------------------------------------------------------
    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            # infos 是 VecEnv step 返回的 info 列表
            infos = self.locals["infos"]

            for info in infos:
                if "reward_terms" not in info:
                    continue

                terms = info["reward_terms"]

                # ====== 奖励项（图 1) 按照reward_compute.py写======
                self.logger.record("reward/R1", terms["R1"])
                self.logger.record("reward/R2", terms["R2"])
                self.logger.record("reward/R3", terms["R3"])
                self.logger.record("reward/R4", terms["R4"])
                self.logger.record("reward/x_dist", terms["x_dist"])
                self.logger.record("reward/y_dist", terms["y_dist"])
                self.logger.record("reward/base_x", terms["base_x"])
                self.logger.record("reward/max_z", terms["max_z"])
                self.logger.record("reward/height", terms["height"])
                self.logger.record("reward/upright", terms["upright"])
                self.logger.record("reward/reward", terms["reward"])
                # self.logger.record("reward/r_1", terms["r1"])
                # self.logger.record("reward/r_2", terms["r2"])
                # self.logger.record("reward/r_4", terms["r4"])
                # self.logger.record("reward/r_5", terms["r5"])
                # self.logger.record("reward/total", terms["reward"])
                # self.logger.record("reward/delta_x", terms["delta_x"])
                # self.logger.record("reward/r8", terms["r8"])
                # self.logger.record("reward/r9", terms["r9"])
                # self.logger.record("reward/r10", terms["r10"])
                # self.logger.record("reward/x_displacement", terms["x_displacement"])
                # self.logger.record("reward/r11", terms["r11"])
                # self.logger.record("reward/upright", terms["upright"])
                # self.logger.record("reward/r12", terms["r12"])
                # self.logger.record("reward/height", terms["height"])
                # self.logger.record("reward/r13", terms["r13"])
                # self.logger.record("reward/r14", terms["r14"])
                # self.logger.record("reward/r15", terms["r15"])
                # self.logger.record("reward/r16", terms["r16"])
                # self.logger.record("reward/r17", terms["r17"])
                # self.logger.record("reward/r18", terms["r18"])
                # ====== 惩罚项（图 2）======
                # self.logger.record("penalty/p_sign", terms["p_sign"])
                # self.logger.record("penalty/p_energy", terms["p_energy"])
                # self.logger.record("penalty/p_action", terms["p_action"])
                # self.logger.record("penalty/p_stall", terms["p_stall"])

                
                
            # for info in infos:
            #     if "original_reward" in info:
            #         # 将原始 reward 写入 TensorBoard（单独命名空间）
                        # self.model.logger.record(
            #             "custom/original_reward",
            #             info["original_reward"]
            #         )
            return True

    reward_callback = RewardCallback()

    # ---------------------------------------------------------
    # 开始训练
    # ---------------------------------------------------------
    model.learn(
        total_timesteps=1000000,          # 训练总步数
        callback=[checkpoint_callback, reward_callback],
        tb_log_name="run1",
        progress_bar=True
    )

    # ---------------------------------------------------------
    # 保存模型与归一化参数
    # ---------------------------------------------------------
    model.save("ppo_hexapod_final")
    # envs.save("vec_normalize.pkl")

    print("Training finished! Model saved as ppo_hexapod_final.zip")
