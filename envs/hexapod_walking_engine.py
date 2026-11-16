import numpy as np
from math import sin, cos, atan2, acos, sqrt, pi

PI = np.pi
EPSILON = 1e-6

ALL_LEGS = np.array([True]*6)

TRIPOD_GAIT = np.array([
    [True,  False, True,  False, True,  False],
    [False, True,  False, True,  False, True ]
])
RIPPLE_GAIT = np.array([
    [True,  False, False, False, True,  False],  # 相1: 抬 leg0 和 leg3
    [False, False, True,  False, False, False],  # 相2: 抬 leg2
    [False, True,  False, False, False, True ],  # 相3: 抬 leg1 和 leg5
    [True,  False, False, False, True,  False],  # 相4: 重复相1（可扩展更多相）
    [False, False, True,  False, False, False],
    [False, True,  False, False, False, True ],
])
#限制 x 的值在 lo 和 hi 之间
def clip(x, lo, hi):
    return max(lo, min(hi, x))
#对给定的布尔数组 mask 进行取反（即将 True 变为 False，反之亦然）
def mask_invert(mask):
    return np.logical_not(mask)

def pose_copy(src):
    return np.array(src, dtype=float)

#将给定 pose 中的每个坐标平移 (x, y, z)，但只有在 mask[i] == True 时才会改变。
def pose_translate(pose, mask, x, y, z):
    for i in range(6):
        if mask[i]:
            pose[i] += np.array([x, y, z])

#对每个坐标点进行旋转操作，旋转的角度由给定的 yaw, pitch, roll 来确定。旋转是绕着 cx, cy, cz 为中心进行的。
def pose_rotate(pose, mask, yaw, pitch, roll, cx, cy, cz):
    cyaw, syaw = cos(yaw), sin(yaw)
    cp, sp = cos(pitch), sin(pitch)
    cr, sr = cos(roll), sin(roll)

    R = np.array([
        [ cyaw*cr, cyaw*sr*sp - syaw*cp, cyaw*sr*cp + syaw*sp],
        [ syaw*cr, syaw*sr*sp + cyaw*cp, syaw*sr*cp - cyaw*sp],
        [   -sr,            cr*sp,             cr*cp]
    ])

    for i in range(6):
        if mask[i]:
            p = pose[i] - np.array([cx, cy, cz])
            pose[i] = R @ p + np.array([cx, cy, cz])
def binom(n, k):
    if k == 0 or k == n:
        return 1
    return binom(n-1, k-1) + binom(n-1, k)

def pose_bezier(dst, points, mask, t):
    n = len(points) - 1
    for i in range(6):
        if mask[i]:
            for j in range(3):
                s = 0.0
                for k in range(n+1):
                    c = binom(n,k)*(1-t)**(n-k)*t**k
                    s += c * points[k][i][j]
                dst[i][j] = s
def no_zero(x):
    return EPSILON if abs(x) < EPSILON else x

# 根据期望的位置计算每只腿的角度。输入参数 rel 是腿部末端相对于身体的位移，而 leg_dim 包含了腿的尺寸（如长度和关节的距离）。
def relative_ik(rel, leg_dim):
    x, y, z = rel
    CX, FM, TB = leg_dim

    r = sqrt(x*x + y*y)
    alpha = -atan2(y, x)
    r -= CX
    d = sqrt(r*r + z*z)

    # 加强 clip，防止超出物理范围
    d = clip(d, 0.0 + EPSILON, FM + TB - EPSILON)  # 避免负数或超上限

    # 计算两个 acos 参数并 clip 到合法范围
    cos_beta1 = (FM*FM + d*d - TB*TB) / (2 * FM * d)
    cos_beta1 = clip(cos_beta1, -1.0, 1.0)

    cos_beta2 = (z*z + d*d - r*r) / (2 * no_zero(z) * d)
    cos_beta2 = clip(cos_beta2, -1.0, 1.0)

    beta = acos(cos_beta1) + acos(cos_beta2)

    cos_gamma = (FM*FM + TB*TB - d*d) / (2 * FM * TB)
    cos_gamma = clip(cos_gamma, -1.0, 1.0)

    gamma = -acos(cos_gamma) + 2*PI
    # print(f"alpha={alpha}, beta={beta}, gamma={gamma}")
    return np.array([
        alpha,
        beta - PI/2,
        gamma - 3*PI/2
    ])

class NikEngineModel:
    def __init__(self):
        self.minimum_command_xy = 0.02 #机器人在 x 和 y 平面上的最小控制命令阈值，小于该值的控制输入将被忽略。
        self.minimum_command_rot = 0.02 #机器人旋转的最小控制命令阈值。

        self.min_step_time = 0.9 #每步的最短持续时间，单位是秒
        self.max_step_time = 1.0 #每步的最长持续时间。
        self.min_step_height = 0.03 #机器人每次抬脚的最低高度。
        self.max_step_height = 1 #机器人每次抬脚的最高高度。

        # self.leg_dim = np.array([[0.065,0.13,0.17]]*6)
        self.leg_dim = np.array([[0.065, 0.13, 0.17 ]]*6)

        self.default_pose = np.array([
            [ 0.20,  0.20, -0.10],
            [ 0.26,  0.00, -0.10],
            [ 0.20, -0.20, -0.10],
            [-0.20, -0.20, -0.10],
            [-0.26,  0.00, -0.10],
            [-0.20,  0.20, -0.10]
        ])

        # self.pose_offset = np.array([
        #     [ 0.0685,  0.0775, 0],
        #     [ 0.093 ,  0     , 0],
        #     [ 0.0685, -0.0775, 0],
        #     [-0.0685, -0.0775, 0],
        #     [-0.093 ,  0     , 0],
        #     [-0.0685,  0.0775, 0]
        # ])
        self.pose_offset = np.array([
            [13.7e-2 / 2, 15.5e-2 / 2, 0],
            [18.6e-2 / 2, 0, 0],
            [13.7e-2 / 2, -15.5e-2 / 2, 0],
            [-13.7e-2 / 2, -15.5e-2 / 2, 0],
            [-18.6e-2 / 2, 0, 0],
            [-13.7e-2 / 2, 15.5e-2 / 2, 0]
        ])
        self.pose_rel_convert = np.array([
            [ 1,  1, 1],
            [ 1,  1, 1],
            [ 1,  1, 1],
            [-1, -1, 1],
            [-1, -1, 1],
            [-1, -1, 1]
        ])

        # self.servo_offset = np.zeros((6,3))
        self.servo_offset = np.array([
            [PI / 4, 0, 0],
            [0, 0, 0],
            [-PI / 4, 0, 0],
            [PI / 4, 0, 0],
            [0, 0, 0],
            [-PI / 4, 0, 0]
        ])
        # self.model_offset = np.array([[0,-1.2734,-0.7854]]*6)
        # self.model_offset = np.array([
        #     [0.5, -2, -0.7854],  # leg 0
        #     [-0.5, -2, -0.7854],  # leg 1
        #     [-0.5, -2, -0.7854],  # leg 2
        #     [ 0.5, -2, -0.7854],  # leg 3
        #     [ 0.5, -2, -0.7854],  # leg 4
        #     [ -0.5, -2, -0.7854],  # leg 5
        # ])
        # self.model_offset = np.zeros((6, 3))
        self.model_offset = np.array([
            [0, -1.2734, -0.7854],
            [0, -1.2734, -0.7854],
            [0, -1.2734, -0.7854],
            [0, -1.2734, -0.7854],
            [0, -1.2734, -0.7854],
            [0, -1.2734, -0.7854],
        ])
       
        

class NikEngineState:
    STAND = 0
    WALK  = 1

    def __init__(self, model):
        self.pose = pose_copy(model.default_pose)
        self.last_step_pose = pose_copy(model.default_pose)
        
        self.yaw = 0.0

        self.mode = self.STAND
        self.prev_mode = self.STAND

        self.step_time = 0.5
        self.step_height = 0.2

        self._step_time = self.step_time
        self._step_height = self.step_height

        self.step_start_time = 0.0
        self.step_progress = 0.0

        self.gait = TRIPOD_GAIT
        self.gait_step = 1   # ★关键：对称启动

# 控制核心 
# model:这是机器人的模型对象，包含了机器人的物理参数、姿势、步态等信息。
# state:这是机器人的当前状态对象，它保存了机器人的当前姿态、步态、模式（如站立、行走等）、步态开始时间、步态进度等。
# t:这是当前时间（时间戳），用于计算步态的进度。例如，用它来确定当前步态的时间进度，控制每步的开始和结束时间。
# cmdvx:这是机器人的目标前进速度（x轴方向）。它决定了机器人在 x 方向上的移动速度。通常是一个浮动值，单位为米每秒（m/s）。
# cmdvy:这是机器人的目标侧向速度（y轴方向）。它决定了机器人在 y 方向上的移动速度。通常是一个浮动值，单位为米每秒（m/s）。
# cmdrot:这是机器人的目标旋转速度（绕 z 轴的角速度）。它决定了机器人绕 z 轴旋转的速度，通常是一个浮动值，单位为弧度每秒（rad/s）。
# dt:这是仿真步长（时间增量），通常是每一帧的仿真时间。它用于控制每次更新时步态和机器人的状态变化的速度。dt 通常是 model.opt.timestep 设定的值，通常为一个小的数值，如 0.01 秒。
def nik_engine_step(model, state, t, cmdvx, cmdvy, cmdrot, dt):

    if state.mode == state.STAND:
        state.pose = pose_copy(model.default_pose)#回到默认姿势
        if (abs(cmdvx) > model.minimum_command_xy or
            abs(cmdvy) > model.minimum_command_xy or
            abs(cmdrot) > model.minimum_command_rot):#超过最小命令阈值，切换到 WALK 模式
            state.mode = state.WALK
            state.step_start_time = t
            # state.gait_step = (state.gait_step + 1) % 2
            state.gait_step = (state.gait_step + 1) % len(state.gait)  # 支持任意相数，同时把 gait_step 推进一格（从奇数相或偶数相开始，避免对称启动问题）
        return

    # ==================== WALK 模式 ====================
    ramp = clip((t - state.step_start_time) / 1.0, 0.0, 1.0) #从站立切换到行走后的前1秒内，把所有速度指令线性从 0 增加到目标值
    cmdvx *= ramp
    cmdvy *= ramp
    cmdrot *= ramp
    state.yaw += cmdrot * dt #累加角速度，得到当前机器人的朝向（弧度）

    # 更新步态进度
    if t - state.step_start_time > state._step_time:
        state.last_step_pose = pose_copy(state.pose)
        state.step_start_time = t
        # state.gait_step = (state.gait_step + 1) % 2
        state.gait_step = (state.gait_step + 1) % len(state.gait)  # 支持任意相数
        state.step_progress = 0.0
    else:
        state.step_progress = (t - state.step_start_time) / state._step_time

    on_ground = mask_invert(state.gait[state.gait_step]) #on_ground：True 表示这条腿当前踩在地上（支撑相）
    stepping  = state.gait[state.gait_step] #stepping：True 表示这条腿正在抬起（摆动相）

    # ============ 关键修改：正确的身体推进方式 ============

    # 1. 支撑腿推动身体前进（速度 = cmdvx, cmdvy）
    #    每帧移动 cmdvx * dt（自然积分）
    pose_translate(state.pose, on_ground, cmdvy * dt, cmdvx * dt, 0)
    # cy = np.cos(state.yaw) #1
    # sy = np.sin(state.yaw) #0

    # dx_world =  cmdvx * cy - cmdvy * sy
    # dy_world =  cmdvx * sy + cmdvy * cy

    # pose_translate(state.pose, on_ground, dy_world * dt, dx_world * dt, 0)


    # # 新增：支撑腿旋转引起的相对划动（产生扭矩）
    # rot_radius = 0.22  # 经验值，根据 default_pose 腿部平均距离调整（0.2~0.25）
    # rot_dy = -cmdrot * dt * rot_radius  # y方向偏移产生偏航（负号取决于旋转方向定义，正/负可调）
    # pose_translate(state.pose, on_ground, rot_dy, 0, 0)  # 只在 y 方向施加，简单有效

    # 2. 迈步腿的目标位置：向前迈一个完整步幅
    #    步幅 = cmdvx * step_time（即在 step_time 内前进的距离）
    # stride_x = cmdvx * state._step_time
    # stride_y = cmdvy * state._step_time
    stride_body_x = cmdvx * state._step_time
    stride_body_y = cmdvy * state._step_time
    # stride_body_y *= 4.0
    # stride_x =  stride_body_x * cy - stride_body_y * sy
    # stride_y =  stride_body_x * sy + stride_body_y * cy
    stride_x = stride_body_x
    stride_y = stride_body_y


    target = pose_copy(model.default_pose)
    pose_translate(target, stepping, -stride_y, -stride_x, 0)

    # ============ 动态限制步高和步时（推荐保留）============
    state._step_time = clip(state.step_time, model.min_step_time, model.max_step_time)
    # max_allowed_height = min(0.08, model.max_step_height * 0.6)
    state._step_height = clip(state.step_height, model.min_step_height, model.max_step_height)
    # print(f"stride_x={stride_x:.3f}, stride_y={stride_y:.3f}")
    # ==================== 贝塞尔曲线轨迹 ====================
    p0 = pose_copy(state.last_step_pose)
    p1 = pose_copy(p0)
    pose_translate(p1, stepping, stride_x * 0.3, stride_y * 0.3, state._step_height * 1.2)  # 前移 + 高抬
    p2 = pose_copy(target)
    pose_translate(p2, stepping, -stride_x * 0.3, -stride_y * 0.3, state._step_height * 1.2)  # 后移 + 高抬
    p3 = pose_copy(target)

    pose_bezier(state.pose, [p0, p1, p2, p3], stepping, state.step_progress)

def state_to_angles(model, state):
    angles = np.zeros(18)

    final = pose_copy(state.pose)
    rel = (final - model.pose_offset) * model.pose_rel_convert

    for i in range(6):
        q = relative_ik(rel[i], model.leg_dim[i])
        # q[0] = q[0] * 4.0
        angles[i*3:(i+1)*3] = q + model.servo_offset[i] + model.model_offset[i]
        # print(f"angles={angles[i*3:(i+1)*3]}")

    return angles