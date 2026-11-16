# import os
# import gymnasium as gym
# from gymnasium import spaces
import numpy as np
# import mujoco
# from collections import deque
# from hexapod_env import HexapodEnv
import mujoco
# env = HexapodEnv(xml_path="plane_scene.xml", render_mode="human")
def compute_reward(env,action):
    reward = 0.0
    r_h = 0.0
    r_v = 0.0
    r_x = 0.0
    r_f = 0.0
    p_sign = 0.0
    p_energy = 0.0
    p_action = 0.0
    p_stall = 0.0
    terminated = False

    # =========================
    # 1. Upright 防翻倒
    # =========================
    # avoid_upright(env)
    r_avoid_upright,base_id, upright = avoid_upright(env)

    # =========================
    # 2. Height
    # =========================
    
    height,r_height_limit,height_term,sign_changes = height_limit(env,base_id)
   
    # =========================
    # 3. 足"端"支撑
    # =========================  
    # r_feet_end = feet_end(env,height)
    # r_foot_slip = foot_slip(env)
    # =========================
    # 4. base的位移
    # =========================
    # r_base_x_disp,sum_plus,dx_total,stall_sign,x_displacement,target_dist,vx,delta_x,xpos = base_x_disp(env,base_id)
    
    # =========================
    # 5.Face target reward
    # =========================
    # r_face_target = face_target(env,base_id,xpos,target_dist)

    #==========================
    # 6. fake_walk
    #==========================
    # r_fake_walk = fake_walk(env)

    #==========================
    # 7. ncon_limit
    #==========================
    r_foot_ncon_limit = foot_ncon_limit(env)
    
    #==========================
    # 8. 强限制
    #==========================
    if height < 0.04:
        print("强限制错误")
        terminated = True
   
    #================奖励项==================
    # r_h = env.reward_value['r_height_term'] * height_term
    # if env.data.ncon >=3 and env.data.ncon == 6:
    #     r_v = env.reward_value['r_vx'] * np.clip(vx, 0.0, 0.05)
    #     # r_v = self.reward_value['r_vx'] * max(vx, 0.0)
    # else:
    #     r_v = 0
    # if delta_x >= 0.001 and delta_x <= 0.005:
    #     r_x = self.reward_value['r_x_displacement'] * delta_x
    
    #================惩罚项==================
    # p_sign = env.reward_value['r_sign_changes'] * sign_changes
    # # 能量（轻）避免机器人“乱动腿”或过度用力
    # p_energy = env.reward_value['r_energy'] * np.sum(env.data.ctrl ** 2)
    # # 动作平滑（很轻）让机器人动作连续平滑，避免关节突然抖动或“抖腿”。
    # p_action = env.reward_value['r_action'] * np.sum((action - env.prev_action) ** 2)
    # p_stall = env.reward_value['r_stall'] * stall_sign

    # reward += r_h + r_v + r_x + r_f
    # reward -= p_sign + p_energy + p_action + p_stall
    # reward += (r_avoid_upright 
    # + r_height_limit 
    # # + r_feet_end 
    # # + r_foot_slip 
    # + r_base_x_disp 
    # + r_face_target
    # # + r_fake_walk
    # + r_foot_ncon_limit
    # )
    # target_dist = env.reward_value['x_goal'] - (xpos - env.init_xpos)
    # reward += 2 * reward_face_target(env, base_id, xpos, target_dist)
    # reward -= 1 * reward_stall(env, target_dist)
    # reward -= 0.000000001 * reward_dof_acc(env)
    # reward -= 0.1 * reward_action_rate(env, action)
    # # reward += 20 * reward_target(env, target_dist)
    # # reward += vx * 10
    # reward += 5.0 * max(vx, 0.0)
    # reward += r_foot_ncon_limit * 5

    # r1 = -2.0 * lin_vel_z_l2(env)
    # r2 = -0.05 * ang_vel_xy_l2(env)
    # # r3 = -0.000025 * joint_torques_l2(env)
    # r4 = -0.00000025 * joint_acc_l2(env)
    # r5 = -1.0 * joint_pos_limits(env, margin=0.05)
    # # r6 = -0.00002 * joint_power(env)
    # # r7 = -0.02 * joint_pos_penalty(env)
    # r8 = -2.0 * stand_still_without_cmd(env, vel_eps=0.05)
    # r9 = -0.3 * action_rate_l2(env, action)
    # r10 = -1.0 * undesired_contacts(env)
    # r11 = -0.0001 * contact_forces(env)
    # r12 = 2.0 * track_lin_vel_xy_exp(env, sigma=0.25)
    # r13 = 1.0 * track_ang_vel_z_exp(env, sigma=0.25)
    # r14 = 0.5 * feet_air_time(env)
    # r15 = 0.2 * feet_contact(env)
    # r16 = 0.1 * feet_contact_without_cmd(env)
    # r17 = 0.1 * feet_height_body(env)
    # r18 = 0.5 * upward(env)
    r_get_body_max_z, max_z = get_body_max_z(env, base_id)
    R1 = r_height_limit
    R2 = r_get_body_max_z
    # R3 = -200.0 * lin_vel_z_l2(env)
    R3 = 0.0
    R4 = -0.0000001 * joint_acc_l2(env)
    # reward = r1 + r2 + r4 + r5 + r8 +r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18
    reward = R1 + R2 + R3
    base_x = env.data.xpos[base_id][0]

    # print("contacts:", env.get_foot_contacts())
    # print(f"r1={r1:.5f},r2={r2:.5f}, r4={r4:.5f},r5={r5:.5f}, "
    #       f" |  r9={r9:.5f}, r10={r10:.5f}, r11={r11:.5f}, r12={r12:.5f}, r13={r13:.5f}, r14={r14:.5f}, r15={r15:.5f}, r16={r16:.5f}, r17={r17:.5f}, r18={r18:.5f}"
    #       f" |  | r={reward:.5f}, upright={upright:.3f}, height={height:.3f},x_displacement = {x_displacement:.5f},height={height:.3f}")
    # print(f"R1={R1:.5f}, R2={R2:.5f}, R3={R3:.5f}, R4={R4:.5f}| r={reward:.5f},  height={height:.3f}, max_z={max_z:.3f}, base_x={base_x:.3f}")
    env.prev_foot_pos_world = env.get_foot_world_positions().copy()
    env.has_prev_foot_pos = True
    xpos = env.data.xpos[env.base_id][0]
    ypos = env.data.xpos[env.base_id][1]  # 获取机器人当前的 y 坐标

    # 目标点的坐标
    target_x = env.reward_value['x_goal']
    target_y = env.reward_value['y_goal']  # 假设y_goal已经定义在reward_value中
    x_dist = target_x - xpos
    y_dist = target_y - ypos
    reward_terms = {
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "R4": R4,
        "base_x": base_x,
        "max_z": max_z,
        "height": height,
        "upright": upright,
        "reward": reward,
        "x_dist": x_dist,
        "y_dist": y_dist,
        # "r1": r1,
        # "r2": r2,
        # "r4": r4,
        # "r5": r5,
        # "reward":reward,
        # "delta_x":delta_x,
        # "x_displacement":x_displacement,
        # "r8":r8,
        # "r9":r9,
        # "r10":r10,
        # "upright":upright,
        # "r11":r11,
        # "r12":r12,
        # "r13":r13,
        # "height": height,
        # "r14": r14,
        # "r15": r15,
        # "r16": r16,
        # "r17": r17,
        # "r18": r18,
    }

    return reward, terminated, reward_terms

def get_body_max_z(env, body_id):
    # """
    # 获取给定 body 的最大 z 高度（考虑所有 geom）。
    
    # 参数：
    #     model: mujoco 模型对象
    #     data: mujoco 仿真数据对象
    #     body_id: body 的 ID（通过 model.body_name2id 获取）
        
    # 返回：
    #     max_z: 该 body 的最高点 z 值
    # """
    # 获取该 body 下的所有 geom
    max_z = env.data.xpos[body_id][2]  # 从 body 参考点开始计算
    # body_geoms = env.model.body_geoms[body_id]  # 获取与 body 相关的所有 geom
    body_geoms = np.where(env.model.geom_bodyid == body_id)[0]
    for geom_id in body_geoms:
        # 获取该 geom 的位置
        geom_pos = env.data.geom_xpos[geom_id]
        # 获取该 geom 的类型和尺寸
        geom_type = env.model.geom_type[geom_id]
        size = env.model.geom_size[geom_id]

        if geom_type == 2:
            # 对于 cylinder 类型 geom，尺寸是半径和高度
            radius = size[0]
            height = size[1]
            # 计算最高和最低 z 坐标
            max_z = max(max_z, geom_pos[2] + height / 2)
            min_z = min(min_z, geom_pos[2] - height / 2)
        
        elif geom_type == 1:
            # 对于 box 类型 geom，尺寸是长宽高
            length = size[0]
            width = size[1]
            height = size[2]
            # 计算最高和最低 z 坐标
            max_z = max(max_z, geom_pos[2] + height / 2)
            min_z = min(min_z, geom_pos[2] - height / 2)

        elif geom_type == 0:
            # 对于 sphere 类型 geom，尺寸是半径
            radius = size[0]
            max_z = max(max_z, geom_pos[2] + radius)
            min_z = min(min_z, geom_pos[2] - radius)
        
        # 根据实际情况，增加更多对其他 geom 类型的处理...

    if max_z >= 0.25:
        r_get_body_max_z = -10.0
    else:
        r_get_body_max_z = 5.0
    return r_get_body_max_z, max_z

def lin_vel_z_l2(env):
    vz = env.data.qvel[2]     # base z 方向线速度
    return vz * vz

def ang_vel_xy_l2(env):
    wx = env.data.qvel[3]
    wy = env.data.qvel[4]
    return wx * wx + wy * wy

def joint_torques_l2(env):
    tau = env.data.ctrl
    return float(np.sum(tau * tau))

def joint_acc_l2(env):
    # qacc 包含 free joint + actuated joint
    qacc = env.data.qacc[:18]  # 只取前 18 个关节（6 个腿，每个腿 3 个关节）
    return float(np.sum(qacc * qacc))

def joint_pos_limits(env, margin=0.05):
    q = env.data.qpos[env.joint_ids]
    qmin = env.joint_qmin
    qmax = env.joint_qmax

    lower_violation = np.clip(qmin + margin - q, 0.0, None)
    upper_violation = np.clip(q - (qmax - margin), 0.0, None)

    return float(np.sum(lower_violation**2 + upper_violation**2))

def joint_power(env):
    tau = env.data.ctrl
    qvel = env.data.qvel[env.joint_ids]
    power = tau * qvel
    return float(np.sum(power * power))

# def joint_pos_penalty(env):
#     q = env.data.qpos[env.joint_ids]
#     q_ref = env.joint_qpos_default   # 站立姿态
#     return float(np.sum((q - q_ref)**2))

def stand_still_without_cmd(env, vel_eps=0.05):
    if env.command_vel_xy is None:
        return 0.0

    if np.linalg.norm(env.command_vel_xy) > vel_eps:
        return 0.0

    vx, vy = env.data.qvel[0], env.data.qvel[1]
    wz = env.data.qvel[5]

    return float(vx*vx + vy*vy + 0.2*wz*wz)

def action_rate_l2(env, action):
    if env.prev_action is None:
        return 0.0
    diff = action - env.prev_action

    return float(np.sum(diff * diff))

def undesired_contacts(env):
    """
    非足端与地面接触惩罚
    """
    bad = 0
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1 = c.geom1
        g2 = c.geom2

        name1 = env.model.geom(g1).name
        name2 = env.model.geom(g2).name

        if "foot" in name1 or "foot" in name2:
            continue
        if "floor" in name1 or "floor" in name2:
            bad += 1

    return bad

def contact_forces(env):
    total = 0.0
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        f = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, i, f)
        total += l2(f[:3])   # 只算线力
    return total

def track_lin_vel_xy_exp(env, sigma=0.25):
    v = env.data.qvel[:2]
    v_cmd = env.command_vel_xy
    err = v - v_cmd
    return np.exp(-l2(err) / sigma)

def track_ang_vel_z_exp(env, sigma=0.25):
    wz = env.data.qvel[5]
    wz_cmd = getattr(env, "command_yaw_rate", 0.0)
    err = wz - wz_cmd
    return np.exp(-(err * err) / sigma)

def feet_air_time(env):
    reward = 0.0
    contacts = env.get_foot_contacts()

    for i in range(6):
        if not contacts[i]:
            env.feet_air_time[i] += env.dt
            reward += env.feet_air_time[i]
        else:
            env.feet_air_time[i] = 0.0

    return reward

def feet_contact(env):
    contacts = env.get_foot_contacts()
    return np.sum(contacts) / 6.0

def feet_contact_without_cmd(env):
    if np.linalg.norm(env.command_vel_xy) > 1e-3:
        return 0.0
    contacts = np.array(env.get_foot_contacts(), dtype=np.float32)
    return float(np.sum(1.0 - contacts))

def feet_height_body(env):
    base_z = env.data.xpos[env.base_id][2]
    foot_pos = env.get_foot_world_positions()
    heights = base_z - foot_pos[:, 2]
    return np.mean(heights)

def upward(env):
    R = env.data.xmat[env.base_id].reshape(3, 3)
    return R[2, 2]   # ∈ [-1, 1]



#朝向目标奖励
def reward_face_target(env, base_id, xpos, target_dist):
    # === robot yaw ===
    R = env.data.xmat[base_id].reshape(3, 3)
    forward = R[:2, 0]
    robot_yaw = np.arctan2(forward[1], forward[0])

    # === target yaw ===
    dx = env.reward_value["x_goal"] - xpos
    dy = 0.0
    target_yaw = np.arctan2(dy, dx)

    # === wrapped angle error ===
    target_angle = np.arctan2(
        np.sin(robot_yaw - target_yaw),
        np.cos(robot_yaw - target_yaw)
    )

    # === reward ===
    if target_dist > 0.12:  
        r = np.exp(
            - (target_angle ** 2) / env.reward_value["target_face_sigma"]
        )
    else:
        r = 1.0

    return r

def reward_stall(env, target_dist):
    # base free joint linear velocity
    vx = env.data.qvel[0] # 机器人x方向线速度
    vy = env.data.qvel[1] # 机器人y方向线速度

    flat_speed = np.sqrt(vx * vx + vy * vy) # 水平面合速度

    if flat_speed < 0.05 and target_dist > 0.2:
        return 1.0
    else:
        return 0.0

def reward_action_rate(env, action):
    if env.prev_action is None:
        return 0.0

    diff = env.prev_action - action
    return np.sum(diff * diff)

def reward_dof_acc(env):
    qacc = env.data.qacc[:18]
    return np.sum(qacc * qacc)

def reward_target(env, target_dist):
    t = env.sim_time
    T = env.reward_value["max_episode_length_s"]
    stay = env.reward_value["target_stay_time"]
    sigma = env.reward_value["target_sigma"]

    if t > T - stay:
        return (1.0 / (stay * (1 + sigma * target_dist ** 2))) * (T / stay)
    else:
        return 0.0


def avoid_upright(env):
    terminated = False
    r_avoid_upright = 0.0
    base_id = env.base_id
    R = env.data.xmat[base_id].reshape(3, 3)
    upright = R[2, 2]
    
    if upright < env.reward_value["upright_limit"]:
        print("错误1")
        # terminated = True
    else:
        r_avoid_upright = 1
        # terminated = True
    # reward += 0.5 * upright
    return r_avoid_upright,base_id, upright

def height_limit(env,base_id):
    r_height_limit = 0.0
    height = env.data.xpos[base_id][2]
    #===========防高度不合理===============
    if height < 0.06 or height > 0.30:
        print("高度不符合")
        r_height_limit = -10
        # terminated = True
    else:
        r_height_limit = 5
    
    # Coxa 0.06, tibia 0.22, femur 0.12
    h_ref = env.reward_value["h_ref"]#微跳
    h_min = env.reward_value["h_min"] #趴着

    #小于0.15 = 0.0，大于0.22 = 1.0
    height_term = np.clip((height - h_min) / (h_ref - h_min), 0.0, 1.0)
    # r_height_limit = 0.5 * height_term
    #===========高度变化频率===============
    env.height_buffer.append(height)
    sign_changes = 0
    # 高度振动计算：高度变化方向变化次数
    if len(env.height_buffer) >= 3:
        diffs = np.diff(env.height_buffer)  # 连续高度差
        signs = np.sign(diffs)               # 方向
        sign_changes = np.sum(signs[1:] != signs[:-1])  # 方向变化次数
        # 振动频率高惩罚更大
        if sign_changes >= env.reward_value["sign_changes_limit"]:
            print("错误3")
            # terminated = True
 
    # height_ok = (height < h_ref and height > h_min)
    # height_sign = 0
    # if height_ok:
    #     height_sign = 1
    # posture = upright * height_term
    # reward += self.reward_value["r_height_term"] * height_term # 给一点站姿奖励，帮助早期学习
    return height, r_height_limit,height_term,sign_changes

def feet_end(env,height):
    r_feet_end = 0.0
    foot_names = [f"leg_{i}_foot" for i in range(1, 7)]
    foot_ids = [env.model.site(name).id for name in foot_names]
    feet_pos = [env.data.site_xpos[i] for i in foot_ids]
    feet_x = np.array([pos[0] for pos in feet_pos])
    delta_feet_x = feet_x - env.prev_feet_x
    forward_swing = np.clip(delta_feet_x, 0.0, 0.01)
    if height > 0.10:
        r_feet_end = 5.0 * np.sum(forward_swing)
    else:
        r_feet_end = -0.5
    env.prev_feet_x = feet_x.copy()
    sum_delta_feet_x = np.sum(delta_feet_x)
    #  if sum_delta_feet_x > 0:
    #     reward += sum_delta_feet_x
    # ground_z = 0.0
    # tol = 0.09  # 1cm 容差
    # num_feet_on_ground = sum(1 for pos in feet_pos if pos[2] < ground_z + tol)

    # 奖励：更多支撑腿 -> 更稳定
    # support_reward = num_feet_on_ground / 6.0
    # if num_feet_on_ground <= 2:
    #     print("error num_feet")
    #     reward -= 2.0
    #     terminated = True
    # reward -= 6.0 * support_reward *2.0

    # test
    r_feet_end = 0.0
    return r_feet_end

def base_x_disp(env,base_id):
    r_base_x_disp = 0.0
    xpos = env.data.xpos[base_id][0]
    delta_x = xpos - env.prev_xpos
    vx = (xpos - env.prev_xpos) / env.dt
    env.prev_xpos = xpos
    # reward += self.reward_value["r_vx"] * np.tanh(vx) #平滑处理
    flat_speed = abs(vx)
    x_displacement = xpos - env.init_xpos
    dx_total = x_displacement - env.prev_x_disp
    env.prev_x_disp = x_displacement
    # ============停滞==========
    target_dist = abs(env.reward_value["x_goal"] - x_displacement)
    stall_sign = 0
    if flat_speed < env.reward_value["flat_speed_limit"] and target_dist > env.reward_value["target_dist_limit"]:
        stall_sign = 1
    r_base_x_disp += 4.0 * max(delta_x, 0.0) 
    # 腿的动作更多
    # leg_movement = np.sum(np.abs(action))  
    # reward += 0.1 * leg_movement
    # if dx_total >=0:
    #     env.x_disp_plus_buffer.append(dx_total)
    # else:
    #     env.x_disp_sub_buffer.append(-dx_total)
    sum_plus = sum(env.x_disp_plus_buffer)
    # sum_minus = sum(env.x_disp_sub_buffer)
    # if sum_plus > 1.2 * sum_minus and sum_plus > 0.01:
    #     r_base_x_disp += sum_plus * 20
    # else:
    #     r_base_x_disp += -0.5

    # if dx_total >= 0.0002:
    #     r_base_x_disp += env.reward_value['r_x_displacement'] * np.clip(dx_total, 0.001, 0.003)
    # # if x_displacement >= 0.04:
    # #     reward += x_displacement * 20
    # else:
    #     r_base_x_disp += -0.1
    # if x_displacement > 0.04:
    #     r_base_x_disp += 10 * np.clip(x_displacement, 0.015, 4.5)
    # else:
    #     r_base_x_disp += -1
    # self.x_disp_buffer.append(dx_total)
    # sign_changes_x_disp = 0
    # if len(self.x_disp_buffer) >= 3:
    #     diffs_x_disp = np.diff(self.x_disp_buffer)
    #     signs_x_disp = np.sign(diffs_x_disp)
    #     sign_changes_x_disp = np.sum(signs_x_disp[1:] != signs_x_disp[:-1])
    #     if sign_changes_x_disp >= self.reward_value["sign_changes_x_disp_limit"]:
    #         print("错误！！！！！！！！！！！！！")
            # reward -= 3
            # terminated = True
    # reward -= self.reward_value["r_stall"] * target_dist
    return r_base_x_disp,sum_plus,dx_total,stall_sign,x_displacement,target_dist,vx,delta_x,xpos

def face_target(env,base_id,xpos,target_dist):
    r_face_target = 0.0
    R = env.data.xmat[base_id].reshape(3, 3)
    forward = R[:2, 0]
    robot_yaw = np.arctan2(forward[1], forward[0])

    dx = env.reward_value["x_goal"] - xpos
    dy = 0.0
    target_yaw = np.arctan2(dy, dx)

    target_angle = np.arctan2(
        np.sin(robot_yaw - target_yaw),
        np.cos(robot_yaw - target_yaw)
    )

    if target_dist > env.reward_value["target_face_dist"]:
        r_face_target = np.exp(
            - (target_angle ** 2) / env.reward_value["target_face_sigma"]
        )
    else:
        r_face_target = 1.0
    r_face_target = env.reward_value["r_face_target"] * r_face_target
    return r_face_target

def foot_slip(env):
    # """
    # 接触足端在世界系发生位移的惩罚
    # """
    if not env.has_prev_foot_pos:
        return 0.0

    contacts = env.get_foot_contacts()
    curr_pos = env.get_foot_world_positions()

    r_foot_slip = 0.0
    for i in range(6):
        if contacts[i]:   # <<< 就是你问的 if foot_in_contact
            diff = curr_pos[i] - env.prev_foot_pos_world[i]
            r_foot_slip += diff[0]**2 + diff[1]**2   # 只惩罚水平滑动
    r_foot_slip = - env.reward_value["r_foot_slip"] * r_foot_slip
    return r_foot_slip

def fake_walk(env):
    # """
    # Base 在水平移动，但接触足几乎不动的惩罚
    # 用于防止“数值滑行 / 假走路”
    # """
    if not env.has_prev_foot_pos:
        return 0.0

    # =====================
    # 1. base 水平速度
    # =====================
    # xpos = env.data.xpos[env.base_id][0]
    # ypos = env.data.xpos[env.base_id][1]

    # prev_x = env.prev_xpos
    # prev_y = getattr(env, "prev_ypos", ypos)

    # v_base_xy = np.sqrt((xpos - prev_x)**2 + (ypos - prev_y)**2) / env.dt

    # env.prev_ypos = ypos  # 缓存
    curr_xy = env.data.xpos[env.base_id][:2]
    prev_xy = env.prev_base_xy.copy()

    v_base_xy = np.linalg.norm(curr_xy - prev_xy) / env.dt
    env.prev_base_xy[:] = curr_xy
   
    # =====================
    # 2. 接触足位移和
    # =====================
    contacts = env.get_foot_contacts()
    curr_pos = env.get_foot_world_positions()

    foot_motion_sum = 0.0
    for i in range(6):
        if contacts[i]:
            diff = curr_pos[i] - env.prev_foot_pos_world[i]
            foot_motion_sum += np.linalg.norm(diff[:2])
    # 阈值 ε
    if v_base_xy < env.reward_value["base_speed_eps"]:
        return 0.0

    # 阈值 δ
    if foot_motion_sum > env.reward_value["foot_motion_min"]:
        return 0.0

    # =====================
    # 3. 条件惩罚
    # =====================
    r_fake_walk = - env.reward_value["r_fake_walk"]

    return r_fake_walk

def foot_ncon_limit(env):
    r_foot_ncon_limit = 0.0
    contacts = env.get_foot_contacts()
    if sum(contacts) < 3:
        r_foot_ncon_limit = -2.0
    else:
        r_foot_ncon_limit = 2.0
    return r_foot_ncon_limit

def l2(x):
    return np.sum(x * x)
