import os
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from collections import deque
from .reward_compute import compute_reward
from .hexapod_walking_engine import NikEngineModel, NikEngineState, nik_engine_step, state_to_angles

class HexapodEnv(gym.Env):
    """
    六足机器人强化学习环境（MuJoCo + Gymnasium）

    设计目标：
    - 稳定站立（不翻倒）
    - 以目标速度向前行走
    - 动作平滑、能耗合理
    - 防止六条腿互相缠绕/碰撞

    本环境可直接用于 PPO / SAC / TD3 等连续控制算法
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # ============================================================
    # 1. 初始化函数
    # ============================================================
    def __init__(self, xml_path="narrow_scene.xml", render_mode=None):
        """
        初始化环境

        参数:
        - xml_path: 六足机器人 MuJoCo XML 模型路径
        - render_mode: 是否启用可视化（human / None）
        """
        super().__init__()

        # ------------------------------------------------------------
        # 1.1 处理 XML 路径（支持相对路径）
        # ------------------------------------------------------------
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(
                os.path.dirname(__file__),
                "assets",
                "hexapod_mujoco2",
                xml_path,
            )

        # ------------------------------------------------------------
        # 1.2 加载 MuJoCo 模型与数据
        # model: 静态结构（关节、质量、几何体）
        # data : 动态状态（位置、速度、接触、力）
        # ------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # print("1111")
        self.data = mujoco.MjData(self.model)
        # ========缓存 geom id=========
        self._cache_geom_ids()
        self.render_mode = render_mode
        self.viewer = None  # ⚠ SubprocVecEnv 中不能使用 viewer
        self.renderer = None
        if self.render_mode == "rgb_array":
            # import mujoco
            self.renderer = mujoco.Renderer(
                self.model,
                width=640,
                height=480
            )
        # 初始化渲染器
        # 创建摄像头并调整视角
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 2.0          # 视角距离机器人，距离越小，机器人越大
        self.cam.elevation = -45         # 俯视角度（负值为俯视，正值为仰视）
        self.cam.azimuth = 0            # 水平角度
        # ---------------------------------------------------------
        # Joint indices and limits (for reward & safety)
        # ---------------------------------------------------------
        self.joint_names = [
            # leg 1
            "leg1coxa", "leg1femur", "leg1tibia",
            # leg 2
            "leg2coxa", "leg2femur", "leg2tibia",
            # leg 3
            "leg3coxa", "leg3femur", "leg3tibia",
            # leg 4
            "leg4coxa", "leg4femur", "leg4tibia",
            # leg 5
            "leg5coxa", "leg5femur", "leg5tibia",
            # leg 6
            "leg6coxa", "leg6femur", "leg6tibia",
        ]

        self.joint_ids = []
        self.joint_ranges = []

        for jname in self.joint_names:
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jname
            )
            assert jid != -1, f"Joint not found: {jname}"

            self.joint_ids.append(self.model.jnt_qposadr[jid])
            self.joint_ranges.append((
                self.model.jnt_range[jid][0],
                self.model.jnt_range[jid][1]
            ))

        self.joint_ids = np.array(self.joint_ids, dtype=np.int32)
        self.joint_ranges = np.array(self.joint_ranges, dtype=np.float32)

        # ------------------------------------------------------------
        # 1.3 动作空间（Action Space）
        # ------------------------------------------------------------
        # nu = 执行器数量（电机/关节）
        # act_dim = self.model.nu
        # self.action_space = spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(act_dim,),
        #     dtype=np.float32,
        # )
        self.action_scale = 0.05  # ★ 小！非常重要
        # 18 个动作：每条腿 3 个关节的增量控制
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(18,),
            dtype=np.float32
        )


        # ------------------------------------------------------------
        # 1.4 观测空间（Observation Space）
        # ------------------------------------------------------------
        # qpos: 所有关节 + 机体位置
        # qvel: 所有关节 + 机体速度
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 仿真时间步长
        self.dt = self.model.opt.timestep

        # ------------------------------------------------------------
        # 1.5 足端 site id（用于防止腿部缠绕）
        # 要求 XML 中存在 leg_1_foot ~ leg_6_foot
        # ------------------------------------------------------------
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"leg_{i}_foot")
            for i in range(1, 7)
        ]

        # ------------------------------------------------------------
        # 1.6 论文常用参数
        # ------------------------------------------------------------
        # 目标前进速度（velocity tracking）
        self.target_vx = 0.3

        # 上一时刻动作（用于平滑性惩罚）
        # self.prev_action = np.zeros(self.model.nu)
        self.prev_action = np.zeros(18, dtype=np.float32)


        # 上一时刻 base x 位置
        self.prev_xpos = 0.0

        # 查询 base id
        self.base_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link"
        )
        
        #=================ik===================
        self.nik_model = NikEngineModel()
        self.nik_state = NikEngineState(self.nik_model)
        # self.nik_state.reset() 
        self.sim_time = 0.0
        for _ in range(5):
            nik_engine_step(
                self.nik_model,
                self.nik_state,
                self.sim_time,
                cmdvx=0.0,
                cmdvy=0.0,
                cmdrot=0.0,
                dt=self.dt
            )
            self.sim_time += self.dt
        # 存储最近 10 个高度
        self.height_buffer = deque(maxlen=10) 
        self.x_disp_buffer = deque(maxlen=20) 
        self.x_disp_plus_buffer = deque(maxlen=20) 
        self.x_disp_sub_buffer = deque(maxlen=20) 

        self.decimation = 4 
        self.reward_value = {
        #=======初始化========
            "h_ref":               0.25,#微跳
            "h_min":               0.15,#趴着
            "x_goal":                 0.0,
            "y_goal":                 -20.0,
            "target_dist_limit":      0.2,
            "target_face_dist":        0.2,
            "target_face_sigma":      0.6,#(0.5~1)
            "vx_limit":               0.02,
            "tracking_gamma_forward": 0.3,
            "tracking_gamma_lateral": 0.3,
            "tracking_gamma_yaw":     4.0,
            "tracking_alpha":         1.0,
            "tracking_beta":          2.0,
            "max_ang_vel":          0.6,
            "target_face_sigma":     0.5,
            "max_episode_length_s":     5.0,
            "target_stay_time":      1,
            "target_sigma":        20,
        #========限制=========
            "upright_limit":        0.6,
            "heightmin_limit":         0.12,
            "heightmax_limit":        0.35,
            "sign_changes_limit":     999,
            "flat_speed_limit":      0.01,
            "sign_changes_x_disp_limit": 15,
            "max_x_limit":        0.3,
            "max_y_limit":        0.3,
            "max_vel":              0.3,
        #=======奖励==========
        # r_h = self.reward_value['r_height_term'] * height_term
        # r_v = self.reward_value['r_vx'] * vx
        # r_x = self.reward_value['r_x_displacement'] * np.clip(delta_x, 0.0, 0.05)
        # r_f = self.reward_value['r_face_target'] * face_reward
            "r_height_term":          0.0,#高度限制
            "r_vx":                 0.0,   #vx
            "r_x_displacement":       100,
            "r_face_target":           2.0,
        #=======惩罚==========
        #  p_sign = self.reward_value['r_sign_changes'] * sign_changes
        # # 能量（轻）避免机器人“乱动腿”或过度用力
        # p_energy = self.reward_value['r_energy'] * np.sum(self.data.ctrl ** 2)
        # # 动作平滑（很轻）让机器人动作连续平滑，避免关节突然抖动或“抖腿”。
        # p_action = self.reward_value['r_action'] * np.sum((action - self.prev_action) ** 2)
        # p_stall = self.reward_value['r_stall'] * stall_sign
            "r_sign_changes":       0.0,
            "r_energy":            0.00,
            "r_action":            0.00,
            "r_stall":                0.0,
            "r_foot_slip":           1.0,
            "r_fake_walk":           0.5,
        #============阈值=============
            "base_speed_eps":         0.001,
            "foot_motion_min":        0.001,
        } 



    # ============================================================
    # 2. reset：环境重置
    # ============================================================
    def reset(self, seed=None, options=None):
        """
        每一回合开始时调用
        - 清空状态
        - 设置机器人初始姿态
        """
        
        super().reset(seed=seed)

        # 重置 MuJoCo 数据（位置、速度、接触等）
        mujoco.mj_resetData(self.model, self.data)

        # 设置机器人初始位置（防止一开始就倒）
        self.data.qpos[0] = 0.0   # x
        self.data.qpos[1] = 0.0   # y
        self.data.qpos[2] = 0.2  # z（高度）

        # 重置上一动作
        self.prev_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        #======关键初始化=======
        self.prev_x_disp = 0.0
        base_id = self.model.body("base_link").id
        self.init_xpos = float(self.data.xpos[base_id][0])  
        self.prev_xpos = self.data.xpos[base_id][0]
        self.prev_height = float(self.data.xpos[base_id][2])
        self.prev_feet_x = np.zeros(6)
        self.prev_foot_pos_world = self.get_foot_world_positions()
        self.prev_base_xy= self.data.xpos[self.base_id][:2].copy()
        self.has_prev_foot_pos = True
        self.height_buffer.clear()
        self.x_disp_buffer.clear()
        self.x_disp_plus_buffer.clear()
        self.x_disp_sub_buffer.clear()
        self.feet_air_time = np.zeros(6, dtype=np.float32)
        self.command_vel_xy = np.zeros(2, dtype=np.float32)

        obs = self._get_obs()
        info = {}
        # print("reset height =", self.data.xpos[self.base_id][2])

        print("[HexapodEnv] reset called")
        return obs, info
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # ============================================================
    # 3. step：执行一步仿真
    # ============================================================
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # 机器人当前坐标
        xpos = self.data.xpos[self.base_id][0]
        ypos = self.data.xpos[self.base_id][1] 

        # 目标点的坐标
        target_x = self.reward_value['x_goal']
        target_y = self.reward_value['y_goal']  # 假设y_goal已经定义在reward_value中
        x_dist = target_x - xpos
        y_dist = target_y - ypos
        # print(f"x_dist: {x_dist:.3f}, y_dist: {y_dist:.3f}")

        # 目标点和当前位置的距离
        target_dist = np.sqrt((target_x - xpos) ** 2 + (target_y - ypos) ** 2)

        # 机器人当前朝向（角度单位）
        R = self.data.xmat[self.base_id].reshape(3, 3)
        forward = R[:2, 0]
        robot_yaw = np.arctan2(forward[1], forward[0])

        # 计算理想的朝向（角度单位）
        dx = target_x - xpos
        dy = target_y - ypos
        target_yaw = np.arctan2(dy, dx)

        # 计算当前朝向和理想朝向的偏差（换算成pai单位）
        target_angle = np.arctan2(
            np.sin(robot_yaw - target_yaw),
            np.cos(robot_yaw - target_yaw)
        )

        # 机器人当前朝向（pai单位）
        desired_yaw = 0.0
        target_pose = np.arctan2(
            np.sin(robot_yaw - desired_yaw),
            np.cos(robot_yaw - desired_yaw)
        )

        # 计算机器人的前进和横向的距离差
        e_forward = float(-target_dist * np.cos(-target_angle))
        e_lateral = float(-target_dist * np.sin(-target_angle))

        # 控制速度
        v_forward = self.reward_value["tracking_gamma_forward"] * e_forward
        v_lateral = self.reward_value["tracking_gamma_lateral"] * e_lateral
        v_forward = np.clip(v_forward, -self.reward_value["max_x_limit"], self.reward_value["max_x_limit"])
        v_lateral = np.clip(v_lateral, -self.reward_value["max_y_limit"], self.reward_value["max_y_limit"])

        # 判断是否超速（对角线速度限速）
        vel_norm = np.sqrt(v_forward * v_forward + v_lateral * v_lateral)
        if vel_norm > self.reward_value["max_vel"]:
            scale = self.reward_value["max_vel"] / vel_norm
            v_forward *= scale
            v_lateral *= scale

        # 计算目标角度的误差
        k = self.reward_value["tracking_alpha"] * np.tanh(target_dist * self.reward_value["tracking_beta"])
        etheta = k * target_angle + (1 - k) * target_pose
        angular_velocity = np.clip(etheta * self.reward_value["tracking_gamma_yaw"], -self.reward_value["max_ang_vel"], self.reward_value["max_ang_vel"])

        # 速度减速因子
        normalized_error = np.clip(abs(etheta) / np.pi, 0.0, 1.0)
        speed_reduction = pow(1.0 - normalized_error, self.reward_value["tracking_beta"])
        v_forward *= speed_reduction
        v_lateral *= speed_reduction

        self.command_vel_xy = np.array([v_forward, v_lateral], dtype=np.float32)

        if target_dist < 0.1 and abs(target_pose) < 0.25:
            v_forward = 0
            v_lateral = 0
            angular_velocity = 0
        # print(f"Step target_dist: {target_dist:.3f}, v_forward: {v_forward:.3f}, v_lateral: {v_lateral:.3f}, angular_velocity: {angular_velocity:.3f},xpos:{xpos:.3f},ypos:{ypos:.3f}")
        
        nik_engine_step(
            self.nik_model,
            self.nik_state,
            self.sim_time,
            cmdvx=v_forward,
            cmdvy=v_lateral,
            cmdrot=angular_velocity,
            dt=self.dt
        )

        q_ref = state_to_angles(self.nik_model, self.nik_state)

        # 更新关节控制
        delta_q = action * self.action_scale
        q_final = q_ref + delta_q
        self.joint_qmin, self.joint_qmax = self.build_joint_limits()
        q_cmd = np.clip(q_final, self.joint_qmin, self.joint_qmax)

        for _ in range(self.decimation):
            self.data.ctrl[:] = q_cmd
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt / self.decimation

        obs = self._get_obs()

        reward, terminated, reward_terms = compute_reward(self, action)
        truncated = False
        self.prev_action = action.copy()

        info = {"reward_terms": reward_terms
                }
        return obs, reward, terminated, truncated, info

    
    def build_joint_limits(self):
        qmin = []
        qmax = []
        JOINT_LIMITS = {
            "coxa":  (-0.6,  0.6),
            "femur": (-1.0,  0.6),
            "tibia": (-1.6, -0.2),
        }
        for _ in range(6):
            qmin.extend([
                JOINT_LIMITS["coxa"][0],
                JOINT_LIMITS["femur"][0],
                JOINT_LIMITS["tibia"][0],
            ])
            qmax.extend([
                JOINT_LIMITS["coxa"][1],
                JOINT_LIMITS["femur"][1],
                JOINT_LIMITS["tibia"][1],
            ])
        return np.array(qmin), np.array(qmax)

    

    # ============================================================
    # 4. 观测函数
    # ============================================================
    def _get_obs(self):
        """
        返回给智能体的状态向量
        """
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    #=========================缓存 geom id=========================    
    def _cache_geom_ids(self):
        # 6 条腿的 tibia geom（真正与地面接触的几何体）
        self.foot_geom_ids = []
        for i in range(1, 7):
            geom_name = f"leg_{i}_tibia_geom"
            geom_id = mujoco.mj_name2id( #根据名称查找对象的ID，返回-1表示未找到
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name
            )
            if geom_id == -1:
                raise RuntimeError(f"Geom not found: {geom_name}")
            self.foot_geom_ids.append(geom_id)

        # 地面 geom
        self.floor_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor"
        )
        if self.floor_geom_id == -1:
            raise RuntimeError("Geom not found: floor")

    #============判断腿末端是否与地面接触================
    def get_foot_contacts(self):
        """
        返回长度为 6 的列表:
        True  -> 该腿末端(=tibia)正在与地面接触
        False -> 未接触
        """
        contacts = [False] * 6

        # 遍历当前时间步所有接触
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2

            # 检查是否是 tibia <-> floor 的接触
            for leg_idx, foot_geom in enumerate(self.foot_geom_ids):
                if ((g1 == foot_geom and g2 == self.floor_geom_id) or
                    (g2 == foot_geom and g1 == self.floor_geom_id)):
                    contacts[leg_idx] = True

        return contacts

    #=============获得腿末端的世界坐标=================
    def get_foot_world_positions(self):
        """
        返回 shape = (6, 3)，每条腿末端的世界坐标
        """
        foot_pos = np.zeros((6, 3))
        for i, geom_id in enumerate(self.foot_geom_ids):
            foot_pos[i] = self.data.geom_xpos[geom_id].copy()
        return foot_pos


    # ============================================================
    # 7. 渲染函数（仅 human 模式）
    # ============================================================
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                print("[ERROR] renderer is None")
                return None  # 防御式，说明 init 出问题
            # 返回 RGB 图像数组
            self.renderer.update_scene(self.data, camera=self.cam)
            frame = self.renderer.render()
            return frame

    # ============================================================
    # 8. 关闭环境
    # ============================================================
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
