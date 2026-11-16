import numpy as np
from math import sin, cos, atan2, acos, sqrt, pi

PI = np.pi
EPSILON = 1e-6

ALL_LEGS = np.array([True]*6)

TRIPOD_GAIT = np.array([
    [True,  False, True,  False, True,  False],
    [False, True,  False, True,  False, True ]
])
def clip(x, lo, hi):
    return max(lo, min(hi, x))

def mask_invert(mask):
    return np.logical_not(mask)

def pose_copy(src):
    return np.array(src, dtype=float)

def pose_translate(pose, mask, x, y, z):
    for i in range(6):
        if mask[i]:
            pose[i] += np.array([x, y, z])

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

def relative_ik(rel, leg_dim):
    x, y, z = rel
    CX, FM, TB = leg_dim

    r = sqrt(x*x + y*y)
    alpha = -atan2(y, x)

    r -= CX
    d = sqrt(r*r + z*z)

    d = clip(d, abs(FM-TB)+EPSILON, FM+TB-EPSILON)

    beta = acos((FM*FM + d*d - TB*TB)/(2*FM*d)) \
         + acos((z*z + d*d - r*r)/(2*no_zero(z)*d))

    gamma = -acos((FM*FM + TB*TB - d*d)/(2*FM*TB)) + 2*PI

    return np.array([
        alpha,
        beta - PI/2,
        gamma - 3*PI/2
    ])
class NikEngineModel:
    def __init__(self):
        self.minimum_command_xy = 0.02
        self.minimum_command_rot = 0.02

        self.min_step_time = 0.3
        self.max_step_time = 1.0
        self.min_step_height = 0.03
        self.max_step_height = 0.15

        self.leg_dim = np.array([[0.065,0.13,0.17]]*6)

        self.default_pose = np.array([
            [ 0.20,  0.20, -0.10],
            [ 0.26,  0.00, -0.10],
            [ 0.20, -0.20, -0.10],
            [-0.20, -0.20, -0.10],
            [-0.26,  0.00, -0.10],
            [-0.20,  0.20, -0.10]
        ])

        self.pose_offset = np.array([
            [ 0.0685,  0.0775, 0],
            [ 0.093 ,  0     , 0],
            [ 0.0685, -0.0775, 0],
            [-0.0685, -0.0775, 0],
            [-0.093 ,  0     , 0],
            [-0.0685,  0.0775, 0]
        ])

        self.pose_rel_convert = np.array([
            [ 1,  1, 1],
            [ 1,  1, 1],
            [ 1,  1, 1],
            [-1, -1, 1],
            [-1, -1, 1],
            [-1, -1, 1]
        ])

        self.servo_offset = np.zeros((6,3))
        # self.model_offset = np.array([[0,-1.2734,-0.7854]]*6)
        self.model_offset = np.array([
            [0.5, -1.2734, -0.7854],  # leg 0
            [-0.15, -1.2734, -0.7854],  # leg 1
            [-0.5, -1.2734, -0.7854],  # leg 2
            [ 0.5, -1.2734, -0.7854],  # leg 3
            [ 0.15, -1.2734, -0.7854],  # leg 4
            [ -0.5, -1.2734, -0.7854],  # leg 5
        ])

class NikEngineState:
    STAND = 0
    WALK  = 1

    def __init__(self, model):
        self.pose = pose_copy(model.default_pose)
        self.last_step_pose = pose_copy(model.default_pose)

        self.mode = self.STAND
        self.prev_mode = self.STAND

        self.step_time = 0.6
        self.step_height = 0.02

        self._step_time = self.step_time
        self._step_height = self.step_height

        self.step_start_time = 0.0
        self.step_progress = 0.0

        self.gait = TRIPOD_GAIT
        self.gait_step = 1   # ★关键：对称启动
def nik_engine_step(model, state, t, cmdvx, cmdvy, cmdrot, dt):

    if state.mode == state.STAND:
        state.pose = pose_copy(model.default_pose)

        if abs(cmdvx) > model.minimum_command_xy \
        or abs(cmdvy) > model.minimum_command_xy \
        or abs(cmdrot) > model.minimum_command_rot:

            state.mode = state.WALK
            state.step_start_time = t
            state.gait_step = (state.gait_step + 1) % 2
        return

    # WALK
    ramp = clip((t - state.step_start_time)/1.0, 0.0, 1.0)
    cmdvx  *= ramp
    cmdvy  *= ramp
    cmdrot *= ramp

    if t - state.step_start_time > state._step_time:
        state.last_step_pose = pose_copy(state.pose)
        state.step_start_time = t
        state.gait_step = (state.gait_step + 1) % 2

    state.step_progress = (t - state.step_start_time)/state._step_time

    on_ground = mask_invert(state.gait[state.gait_step])
    stepping  = state.gait[state.gait_step]

    pose_translate(state.pose, on_ground,
                   cmdvy*dt*2, cmdvx*dt*2, 0)

    target = pose_copy(model.default_pose)
    pose_translate(target, stepping,
                   -cmdvy*state._step_time,
                   -cmdvx*state._step_time, 0)

    p0 = pose_copy(state.last_step_pose)
    p1 = pose_copy(p0); pose_translate(p1, stepping, 0,0,state._step_height)
    p2 = pose_copy(target); pose_translate(p2, stepping, 0,0,state._step_height)
    p3 = pose_copy(target)

    pose_bezier(state.pose, [p0,p1,p2,p3], stepping, state.step_progress)
def state_to_angles(model, state):
    angles = np.zeros(18)

    final = pose_copy(state.pose)
    rel = (final - model.pose_offset) * model.pose_rel_convert

    for i in range(6):
        q = relative_ik(rel[i], model.leg_dim[i])
        angles[i*3:(i+1)*3] = q + model.servo_offset[i] + model.model_offset[i]

    return angles

