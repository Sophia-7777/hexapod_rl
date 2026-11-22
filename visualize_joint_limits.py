import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

# ==========================================
# 1. 加载模型
# ==========================================
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

args = parser.parse_args()
XML_PATH = args.xml

# 限位扇形半径（米）
ARC_RADIUS = 0.045

# 圆弧细分数量，越大越平滑
ARC_SEGMENTS = 40

# 线条宽度（像素）
LINE_WIDTH = 3.0

# 是否将所有关节初始化到 0 rad
RESET_TO_ZERO = False

# 是否显示关节锚点
SHOW_ANCHORS = True


# ============================================================
# 2. 数学工具
# ============================================================

def normalize(v):
    """单位化向量。"""
    norm = np.linalg.norm(v)

    if norm < 1e-10:
        raise ValueError("无法单位化零向量")

    return v / norm


def rotate_vector(v, axis, angle):
    """
    Rodrigues 旋转公式：
    将向量 v 绕 axis 旋转 angle。
    """
    axis = normalize(axis)

    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1 - np.cos(angle))
    )


def get_reference_vectors(axis, body_xmat, q):
    """
    构造关节旋转平面中的两个正交基向量。

    body_xmat: 关节所属 body 的世界旋转矩阵
    axis:      关节世界旋转轴
    q:         当前关节角度

    通过抵消当前关节角度，近似恢复 q=0 时的参考方向。
    """

    axis = normalize(axis)

    R = body_xmat.reshape(3, 3)

    # 选择 body 局部 X 轴作为参考方向
    ref = R[:, 0].copy()

    # 去除当前关节角度的影响
    ref = rotate_vector(ref, axis, -q)

    # 投影到垂直于旋转轴的平面
    ref = ref - np.dot(ref, axis) * axis

    # 如果参考向量与轴接近平行，则换一个方向
    if np.linalg.norm(ref) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0])

        ref = ref - np.dot(ref, axis) * axis

    if np.linalg.norm(ref) < 1e-8:
        ref = np.array([0.0, 0.0, 1.0])

        ref = ref - np.dot(ref, axis) * axis

    u = normalize(ref)

    # 构造旋转平面的第二个基向量
    v = normalize(np.cross(axis, u))

    return u, v


def point_on_arc(anchor, u, v, angle, radius):
    """计算旋转平面内某个角度对应的三维点。"""

    return anchor + radius * (
        np.cos(angle) * u
        + np.sin(angle) * v
    )


# ============================================================
# 3. MuJoCo 自定义几何绘制
# ============================================================

class JointLimitVisualizer:

    def __init__(self, model, data, viewer):

        self.model = model
        self.data = data
        self.viewer = viewer

        # 保存所有有限位的 hinge 关节
        self.joints = []

        self._find_joints()

    def _find_joints(self):

        print("\n========== Joint Limits ==========")

        for jid in range(self.model.njnt):

            # 只处理旋转关节
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue

            # 跳过没有启用限位的关节
            if not self.model.jnt_limited[jid]:
                continue

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid
            )

            if name is None:
                name = f"joint_{jid}"

            qadr = self.model.jnt_qposadr[jid]

            lower = self.model.jnt_range[jid, 0]
            upper = self.model.jnt_range[jid, 1]

            body_id = self.model.jnt_bodyid[jid]

            self.joints.append({
                "id": jid,
                "name": name,
                "qadr": qadr,
                "body_id": body_id,
                "lower": lower,
                "upper": upper,
            })

            print(
                f"{name:20s} | "
                f"Min: {np.rad2deg(lower):8.2f} deg | "
                f"Max: {np.rad2deg(upper):8.2f} deg"
            )

        print(f"\n共找到 {len(self.joints)} 个带限位的旋转关节")
        print("==================================\n")

    # --------------------------------------------------------
    # 绘制一条线段
    # --------------------------------------------------------

    def draw_line(
        self,
        scene,
        index,
        start,
        end,
        color,
        width=LINE_WIDTH
    ):

        geom = scene.geoms[index]

        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array(color, dtype=np.float32)
        )

        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            start,
            end
        )

        return index + 1

    # --------------------------------------------------------
    # 绘制一个球
    # --------------------------------------------------------

    def draw_sphere(
        self,
        scene,
        index,
        position,
        color,
        radius=0.006
    ):

        geom = scene.geoms[index]

        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0, 0]),
            pos=position,
            mat=np.eye(3).flatten(),
            rgba=np.array(color, dtype=np.float32)
        )

        return index + 1

    # --------------------------------------------------------
    # 绘制单个关节的限位扇形
    # --------------------------------------------------------

    def draw_joint(self, scene, index, joint):

        jid = joint["id"]

        lower = joint["lower"]
        upper = joint["upper"]

        qadr = joint["qadr"]
        body_id = joint["body_id"]

        # 当前关节角度
        q = self.data.qpos[qadr]

        # 关节世界坐标锚点
        anchor = self.data.xanchor[jid].copy()

        # 关节世界旋转轴
        axis = normalize(self.data.xaxis[jid].copy())

        # 关节所属 body 的世界旋转矩阵
        body_xmat = self.data.xmat[body_id].copy()

        # 建立旋转平面的参考坐标系
        u, v = get_reference_vectors(
            axis,
            body_xmat,
            q
        )

        radius = ARC_RADIUS

        # ----------------------------------------------------
        # A. 绿色圆弧：允许旋转的角度范围
        # ----------------------------------------------------

        angles = np.linspace(
            lower,
            upper,
            ARC_SEGMENTS + 1
        )

        points = [
            point_on_arc(
                anchor,
                u,
                v,
                angle,
                radius
            )
            for angle in angles
        ]

        for i in range(len(points) - 1):

            index = self.draw_line(
                scene,
                index,
                points[i],
                points[i + 1],
                color=[0.0, 1.0, 0.0, 1.0],
                width=LINE_WIDTH
            )

        # ----------------------------------------------------
        # B. 红色射线：最小角度和最大角度
        # ----------------------------------------------------

        p_min = point_on_arc(
            anchor, u, v, lower, radius
        )

        p_max = point_on_arc(
            anchor, u, v, upper, radius
        )

        index = self.draw_line(
            scene,
            index,
            anchor,
            p_min,
            color=[1.0, 0.0, 0.0, 1.0],
            width=LINE_WIDTH + 1
        )

        index = self.draw_line(
            scene,
            index,
            anchor,
            p_max,
            color=[1.0, 0.0, 0.0, 1.0],
            width=LINE_WIDTH + 1
        )

        # ----------------------------------------------------
        # C. 黄色射线：当前关节角度
        # ----------------------------------------------------

        p_current = point_on_arc(
            anchor, u, v, q, radius
        )

        index = self.draw_line(
            scene,
            index,
            anchor,
            p_current,
            color=[1.0, 1.0, 0.0, 1.0],
            width=LINE_WIDTH + 2
        )

        # ----------------------------------------------------
        # D. 关节中心
        # ----------------------------------------------------

        if SHOW_ANCHORS:

            index = self.draw_sphere(
                scene,
                index,
                anchor,
                color=[1.0, 0.5, 0.0, 1.0]
            )

        return index

    # --------------------------------------------------------
    # 更新所有关节的限位可视化
    # --------------------------------------------------------

    def update(self):

        scene = self.viewer.user_scn

        # 清空上一帧自定义几何
        scene.ngeom = 0

        index = 0

        for joint in self.joints:

            index = self.draw_joint(
                scene,
                index,
                joint
            )

        # 更新自定义几何数量
        scene.ngeom = index


# ============================================================
# 4. 主程序
# ============================================================

def main():

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(XML_PATH)

    data = mujoco.MjData(model)

    # 可选：将关节设置为 0 rad
    if RESET_TO_ZERO:

        for jid in range(model.njnt):

            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:

                qadr = model.jnt_qposadr[jid]

                data.qpos[qadr] = 0.0

    # 更新 MuJoCo 世界状态
    mujoco.mj_forward(model, data)

    # 打开 Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        visualizer = JointLimitVisualizer(
            model,
            data,
            viewer
        )

        print("MuJoCo Joint Limit Visualizer Started")

        while viewer.is_running():

            # 不调用 mj_step：
            # 保持机器人静止，只显示当前关节状态和限位。

            with viewer.lock():

                # 更新自定义限位扇形
                visualizer.update()

            # 同步 Viewer
            viewer.sync()

            time.sleep(0.02)


if __name__ == "__main__":
    main()