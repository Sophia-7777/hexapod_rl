import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import imageio
import subprocess

from pynput import keyboard

os.makedirs("frames", exist_ok=True)

# ====== Engine ======
from envs.hexapod_walking_engine import (
    NikEngineModel,
    NikEngineState,
    nik_engine_step,
    state_to_angles,
)

# ===============================
# MuJoCo model
# ===============================
xml_path = "plane_scene.xml"
if not os.path.isabs(xml_path):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        "envs", "assets", "hexapod_mujoco2",
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
# 键盘状态
# ===============================
keys = {
    'w': False, 'a': False, 's': False, 'd': False,
    'q': False, 'e': False, 'space': False,
}

def _key_name(key):
    try:
        return key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            return 'space'
        return None

def on_press(key):
    k = _key_name(key)
    if k in keys:
        keys[k] = True

def on_release(key):
    k = _key_name(key)
    if k in keys:
        keys[k] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ===============================
# 键盘 → 速度命令
#   W/S : 前进 / 后退     (cmdvx)
#   A/D : 左移 / 右移     (cmdvy)
#   Q/E : 左转 / 右转     (cmdrot)
#   Space: 急停
# ===============================
BASE_VXY  = 0.06   # 平移速度 (m/s)，可调
BASE_VROT = 0.30   # 旋转角速度 (rad/s)，可调

def read_cmd():
    vx = vy = rot = 0.0
    if keys['w']: vx += BASE_VXY
    if keys['s']: vx -= BASE_VXY
    if keys['a']: vy += BASE_VXY
    if keys['d']: vy -= BASE_VXY
    if keys['q']: rot += BASE_VROT
    if keys['e']: rot -= BASE_VROT
    if keys['space']:
        vx = vy = rot = 0.0
    return vx, vy, rot

# ===============================
# 主循环
# ===============================
t = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    # ========= 1. 稳定站立（让自由关节落地）=========
    print("Standing (stabilizing)...")
    for _ in range(300):          # ~3 s
        nik_engine_step(
            eng_model, eng_state,
            t, cmdvx=0.0, cmdvy=0.0, cmdrot=0.0, dt=dt
        )
        q_des = state_to_angles(eng_model, eng_state)
        data.ctrl[:] = q_des
        mujoco.mj_step(model, data)
        viewer.sync()
        t += dt

    # ========= 2. 键盘控制行走 =========
    print("Keyboard control ready: W/S/A/D move, Q/E rotate, Space stop.")
    try:
        while viewer.is_running():
            cmdvx, cmdvy, cmdrot = read_cmd()

            nik_engine_step(
                eng_model, eng_state,
                t,
                cmdvx=cmdvx, cmdvy=cmdvy, cmdrot=cmdrot,
                dt=dt,
            )

            q_des = state_to_angles(eng_model, eng_state)
            data.ctrl[:] = q_des

            mujoco.mj_step(model, data)
            viewer.sync()
            t += dt
    finally:
        listener.stop()