import mujoco
import numpy as np
import imageio
import os 
from tqdm import tqdm

reward_value = {
    #=======初始化========
    "h_ref":               0.25,#微跳
    "h_min":               0.15,#趴着
    "x_goal":                 2.0,
    "y_goal":                 -2.0,
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
    "max_x_limit":        0.1,
    "max_y_limit":        0.1,
    "max_vel":              0.1,
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



# ===============================
# NikEngine
# ===============================
from envs.hexapod_walking_engine import (
    NikEngineModel,
    NikEngineState,
    nik_engine_step,
    state_to_angles
)

# ===============================
# MuJoCo model
# ===============================
# xml_path = "narrow_scene.xml"
xml_path = "plane_scene_point.xml"
if not os.path.isabs(xml_path):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        "envs",
        "assets",
        "hexapod_mujoco2",
        xml_path,
    )

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)
base_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link"
        )
dt = model.opt.timestep


# 1️⃣ 先拿到 body id
target_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "target_marker"
)

# 2️⃣ 由 body id 映射到 mocap id
target_mocap_id = model.body_mocapid[target_body_id]

# ===============================
# Walking engine
# ===============================
eng_model = NikEngineModel()
eng_state = NikEngineState(eng_model)

# ===============================
# Renderer (离屏渲染，高分辨率)
# ===============================
WIDTH = 640
HEIGHT = 480
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# 创建跟随摄像机
cam = mujoco.MjvCamera()
cam.distance = 20.0          # 距离减小 => 机器人更大
cam.elevation = -45          # 俯视角
cam.azimuth = 0

# ===============================
# 仿真时间
# ===============================
t = 0.0

# ===============================
# 核心仿真函数
# ===============================
def sim_step(cmdvx, cmdvy, angular_velocity):
    global t
    data.mocap_pos[target_mocap_id] = np.array([
    reward_value["x_goal"],
    reward_value["y_goal"],
    0.0   # 地面高度，可根据地形改
    ])

    # 更新基础引擎
    nik_engine_step(
        eng_model,
        eng_state,
        t,
        cmdvx=cmdvx,
        cmdvy=cmdvy,
        # cmdrot=cmdrot,
        cmdrot=angular_velocity,
        dt=dt
    )

    # 转换为关节角
    q_des = state_to_angles(eng_model, eng_state)
    data.ctrl[:] = q_des

    # 执行仿真步
    mujoco.mj_step(model, data)
    t += dt

def path(data):
    xpos = data.xpos[base_id][0]
    ypos = data.xpos[base_id][1]  # 获取机器人当前的 y 坐标

    # 目标点的坐标
    target_x = reward_value['x_goal']
    target_y = reward_value['y_goal']  # 假设y_goal已经定义在reward_value中
    x_dist = target_x - xpos
    y_dist = target_y - ypos
    # print(f"x_dist: {x_dist:.3f}, y_dist: {y_dist:.3f}")
    # 目标点和当前位置的距离
    target_dist = np.sqrt((target_x - xpos) ** 2 + (target_y - ypos) ** 2)

    # 旋转矩阵 R，计算机器人当前的朝向
    R = data.xmat[base_id].reshape(3, 3)
    forward = R[:2, 0]
    robot_yaw = np.arctan2(forward[1], forward[0])

    # 计算机器人和目标点的偏角
    dx = target_x - xpos
    dy = target_y - ypos
    target_yaw = np.arctan2(dy, dx)

    # 机器人和目标点的偏角
    target_angle = np.arctan2(
        np.sin(robot_yaw - target_yaw),
        np.cos(robot_yaw - target_yaw)
    )

    # 理想的偏角（目标朝向）
    desired_yaw = 0.0
    target_pose = np.arctan2(
        np.sin(robot_yaw - desired_yaw),
        np.cos(robot_yaw - desired_yaw)
    )

    # 计算前进和横向的误差
    e_forward = float(-target_dist * np.cos(-target_angle))
    e_lateral = float(-target_dist * np.sin(-target_angle))

    # 控制速度
    v_forward = reward_value["tracking_gamma_forward"] * e_forward
    v_lateral = reward_value["tracking_gamma_lateral"] * e_lateral
    v_forward = np.clip(v_forward, -reward_value["max_x_limit"], reward_value["max_x_limit"])
    v_lateral = np.clip(v_lateral, -reward_value["max_y_limit"], reward_value["max_y_limit"])

    # 判断是否超速
    vel_norm = np.sqrt(v_forward * v_forward + v_lateral * v_lateral)
    if vel_norm > reward_value["max_vel"]:
        scale = reward_value["max_vel"] / vel_norm
        v_forward *= scale
        v_lateral *= scale

    # 计算目标角度的误差
    k = reward_value["tracking_alpha"] * np.tanh(target_dist * reward_value["tracking_beta"])
    etheta = k * target_angle + (1 - k) * target_pose
    angular_velocity = np.clip(etheta * reward_value["tracking_gamma_yaw"], -reward_value["max_ang_vel"], reward_value["max_ang_vel"])

    # 速度减速因子
    normalized_error = np.clip(abs(etheta) / np.pi, 0.0, 1.0)
    speed_reduction = pow(1.0 - normalized_error, reward_value["tracking_beta"])
    v_forward *= speed_reduction
    v_lateral *= speed_reduction

    command_vel_xy = np.array([v_forward, v_lateral], dtype=np.float32)

    if target_dist < 0.1 and abs(target_pose) < 0.25:
        v_forward = 0
        v_lateral = 0
        angular_velocity = 0
    print(f"Step target_dist: {target_dist:.3f}, v_forward: {v_forward:.3f}, v_lateral: {v_lateral:.3f}, angular_velocity: {angular_velocity:.3f},xpos:{xpos:.3f},ypos:{ypos:.3f}")
    
    return v_forward, v_lateral 
# ===============================
# 录制视频
# ===============================
frames = []

# -------------------------------#
# 站立阶段
# -------------------------------#
print("Standing phase...")
for _ in range(100):  # 约 0.5 秒
    sim_step(0.0, 0.0, 0.0)

    # 摄像机跟随机器人 base_link
    cam.lookat[:] = data.xpos[0]

    # 渲染
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    frames.append(frame)

# -------------------------------#
# 行走阶段
# -------------------------------#
print("Walking phase...")
for _ in tqdm(range(5000), desc="Walking phase", ncols=100):  # 显示进度条
    # sim_step(0.06, 0.0, 0.0) 
    sim_step(0.0, 0.06, 0.0) 
    # sim_step(0.06, 0.06, 0.0)  
    # sim_step(-0.06, 0.06, 0.0)
    v_forward, v_lateral = path(data)
    v_lateral = -v_lateral
    sim_step(v_forward, v_lateral, 0.0)
    cam.lookat[:] = data.xpos[0]
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    frames.append(frame)
# for _ in range(3000):  # 约 5 秒
#     # sim_step(0.06, 0.0, 0.0) 
#     # sim_step(0.0, 0.06, 0.0) 
#     sim_step(0.0, 0.0, 1.0)  
#     cam.lookat[:] = data.xpos[0]
#     renderer.update_scene(data, camera=cam)
#     frame = renderer.render()
#     frames.append(frame)

# -------------------------------#
# 保存视频
# -------------------------------#
fps = int(1.0 / dt)
output_file = "hexapod_walk.mp4"
print(f"Saving video @ {fps} FPS, frames = {len(frames)}")

imageio.mimsave(
    output_file,
    frames,
    fps=fps,
    codec="libx264"
)

print(f"Video saved as {output_file}")
