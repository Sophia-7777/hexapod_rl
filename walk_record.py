import mujoco
import numpy as np
import imageio
import os 
from tqdm import tqdm

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
xml_path = "narrow_scene.xml"
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

dt = model.opt.timestep

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
cam.distance = 5.0          # 距离减小 => 机器人更大
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
    sim_step(0.0, 0.0, 0.06) 
    # sim_step(0.0, 0.06, 0.0) 
    # sim_step(0.06, 0.06, 0.0)  
    # sim_step(-0.06, 0.06, 0.0)
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
