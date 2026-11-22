"""
Hexapod control engine for MuJoCo.

States:
    SQUAT : crouched joint posture
    STAND : neutral standing joint posture
    WALK  : tripod gait with Cartesian foot targets and MuJoCo-site IK

The engine owns the state machine, gait generator, IK and actuator commands.
Keyboard handling and rendering are kept in simulation.py.
"""

from enum import Enum, auto
import numpy as np
import mujoco


class RobotState(Enum):
    SQUAT = auto()
    STAND = auto()
    WALK = auto()


class SopHexapodGeometry:
    # Order must match leg1 ... leg6 in the XML.
    LEG_ORIGINS = np.array([
        [-0.079498,  0.067171, 0.0],
        [ 0.000000,  0.093000, 0.0],
        [ 0.079498,  0.067171, 0.0],
        [ 0.079498, -0.067171, 0.0],
        [ 0.000000, -0.093000, 0.0],
        [-0.079498, -0.067171, 0.0],
    ], dtype=float)

    GAIT_OFFSETS = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])


class SopHexapodEngine:
    def __init__(
        self,
        model,
        data,
        control_period=0.02,
        transition_duration=0.6,
    ):
        self.model = model
        self.data = data
        self.control_period = float(control_period)
        self.transition_duration = float(transition_duration)

        self.state = RobotState.STAND
        self._pending_state = None
        self._transition_start = None
        self._transition_goal = None
        self._transition_elapsed = 0.0
        self._is_transitioning = False

        # Normalized keyboard command: forward, lateral, yaw.
        self.command = np.zeros(3, dtype=float)

        # Gait parameters. Tune only after the neutral pose is verified.
        self.gait_period = 2.0
        self.stance_ratio = 0.60
        self.step_height = 0.1
        self.max_stride_x = 0.08
        self.max_stride_y = 0.07
        self.max_yaw_stride = 0.025
        self.gait_phase = 0.0

        # Per-leg nominal foot positions in the base_link frame.
        self.nominal_feet = np.zeros((6, 3), dtype=float)

        self.joint_names = [
            f"leg{i}{joint}"
            for i in range(1, 7)
            for joint in ("coxa", "femur", "tibia")
        ]
        self.site_names = [f"leg_{i}_foot" for i in range(1, 7)]

        self.joint_ids = []
        self.qpos_addresses = []
        self.dof_addresses = []
        self.actuator_ids = []
        self.site_ids = []

        for joint_name in self.joint_names:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if jid < 0:
                raise ValueError(f"XML 中找不到关节: {joint_name}")

            aid = np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)
            if len(aid) == 0:
                raise ValueError(f"关节 {joint_name} 没有 position actuator")

            self.joint_ids.append(jid)
            self.qpos_addresses.append(int(self.model.jnt_qposadr[jid]))
            self.dof_addresses.append(int(self.model.jnt_dofadr[jid]))
            self.actuator_ids.append(int(aid[0]))

        for site_name in self.site_names:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if sid < 0:
                raise ValueError(f"XML 中找不到足端 site: {site_name}")
            self.site_ids.append(sid)

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.base_body_id < 0:
            raise ValueError("XML 中找不到 base_link")

        self.joint_ids = np.asarray(self.joint_ids, dtype=int)
        self.qpos_addresses = np.asarray(self.qpos_addresses, dtype=int)
        self.dof_addresses = np.asarray(self.dof_addresses, dtype=int)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=int)
        self.site_ids = np.asarray(self.site_ids, dtype=int)

        # Joint postures are [coxa, femur, tibia] for each of six legs.
        # These are initial tuning values, not a claim of dynamically stable poses.
        self.stand_pose = np.tile(
            np.array([0.0, 0.0, 0.3]), (6, 1)
        )
        self.squat_pose = np.tile(
            np.array([0.0, 0.35, -1.25]), (6, 1)
        )

        self.joint_targets = self.stand_pose.copy()

        self._apply_joint_positions(self.stand_pose)
        mujoco.mj_forward(self.model, self.data)
        self._capture_nominal_feet()

        # Initialize actuators at the same pose as the model.
        self._write_actuators(self.stand_pose)

    # ------------------------------------------------------------------
    # State machine API
    # ------------------------------------------------------------------

    def request_state(self, new_state):
        """Request SQUAT, STAND or WALK. WALK is routed through STAND."""
        if isinstance(new_state, str):
            new_state = RobotState[new_state.upper()]
        if not isinstance(new_state, RobotState):
            raise ValueError("new_state 必须是 RobotState 或状态名称字符串")

        if new_state == RobotState.WALK:
            if self.state == RobotState.SQUAT:
                # Stand first, then enter WALK.
                self._pending_state = RobotState.WALK
                self._begin_transition(self.stand_pose, RobotState.STAND)
                return
            if self.state == RobotState.STAND and not self._is_transitioning:
                self.state = RobotState.WALK
                self._pending_state = None
                self.gait_phase = 0.0
                return
            if self.state == RobotState.WALK:
                return
            return

        self._pending_state = None
        if new_state == RobotState.STAND:
            self.command[:] = 0.0
            self._begin_transition(self.stand_pose, RobotState.STAND)
        elif new_state == RobotState.SQUAT:
            self.command[:] = 0.0
            self._begin_transition(self.squat_pose, RobotState.SQUAT)

    def set_command(self, vx=0.0, vy=0.0, wz=0.0):
        """Set normalized movement command in [-1, 1]."""
        self.command = np.clip(
            np.asarray([vx, vy, wz], dtype=float), -1.0, 1.0
        )

        moving = np.linalg.norm(self.command) > 1e-4
        if moving:
            if self.state == RobotState.STAND and not self._is_transitioning:
                self.request_state(RobotState.WALK)
            elif self.state == RobotState.SQUAT:
                # A walking request from squat automatically stands first.
                self.request_state(RobotState.WALK)
        else:
            if self.state == RobotState.WALK:
                self.request_state(RobotState.STAND)

    def stop(self):
        self.set_command(0.0, 0.0, 0.0)

    def _begin_transition(self, target_pose, final_state):
        self._transition_start = self.joint_targets.copy()
        self._transition_goal = np.asarray(target_pose, dtype=float).copy()
        self._transition_elapsed = 0.0
        self._is_transitioning = True
        self._transition_final_state = final_state

    # ------------------------------------------------------------------
    # Model / actuator helpers
    # ------------------------------------------------------------------

    def _apply_joint_positions(self, angles_6x3):
        self.data.qpos[self.qpos_addresses] = np.asarray(
            angles_6x3, dtype=float
        ).reshape(-1)

    def _write_actuators(self, angles_6x3):
        flat = np.asarray(angles_6x3, dtype=float).reshape(-1)
        self.data.ctrl[self.actuator_ids] = flat

    def _capture_nominal_feet(self):
        base_pos = self.data.xpos[self.base_body_id].copy()
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3).copy()
        for i, sid in enumerate(self.site_ids):
            world_pos = self.data.site_xpos[sid].copy()
            self.nominal_feet[i] = base_rot.T @ (world_pos - base_pos)

    def _joint_limits(self, leg_index):
        ids = self.joint_ids[3 * leg_index: 3 * leg_index + 3]
        lower = self.model.jnt_range[ids, 0]
        upper = self.model.jnt_range[ids, 1]
        return lower, upper

    # ------------------------------------------------------------------
    # MuJoCo-based inverse kinematics
    # ------------------------------------------------------------------

    def _solve_leg_ik(self, leg_index, target_body):
        sid = int(self.site_ids[leg_index])

        qadr = self.qpos_addresses[
            3 * leg_index: 3 * leg_index + 3
        ]

        dof = self.dof_addresses[
            3 * leg_index: 3 * leg_index + 3
        ]

        lower, upper = self._joint_limits(leg_index)

        # 保存真实仿真状态
        original_qpos = self.data.qpos.copy()

        base_pos = self.data.xpos[self.base_body_id].copy()
        base_rot = self.data.xmat[
            self.base_body_id
        ].reshape(3, 3).copy()

        target_world = (
            base_pos
            + base_rot @ np.asarray(target_body, dtype=float)
        )

        q = self.data.qpos[qadr].copy()

        damping = 0.025
        max_iterations = 30
        tolerance = 0.001

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        try:
            for _ in range(max_iterations):

                self.data.qpos[qadr] = q

                mujoco.mj_forward(
                    self.model,
                    self.data
                )

                current = self.data.site_xpos[sid].copy()

                error = target_world - current

                if np.linalg.norm(error) < tolerance:
                    break

                mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jacp,
                    jacr,
                    sid
                )

                J = jacp[:, dof]

                A = (
                    J @ J.T
                    + damping ** 2 * np.eye(3)
                )

                try:
                    dq = J.T @ np.linalg.solve(A, error)
                except np.linalg.LinAlgError:
                    break

                dq_norm = np.linalg.norm(dq)

                if dq_norm > 0.12:
                    dq *= 0.12 / dq_norm

                q = np.clip(
                    q + dq,
                    lower,
                    upper
                )

            # 计算最终足端误差
            self.data.qpos[qadr] = q
            mujoco.mj_forward(self.model, self.data)

            actual = self.data.site_xpos[sid].copy()
            residual = np.linalg.norm(target_world - actual)

        finally:
            # 恢复真实物理状态
            self.data.qpos[:] = original_qpos
            mujoco.mj_forward(self.model, self.data)

        return q

    # ------------------------------------------------------------------
    # Gait trajectory
    # ------------------------------------------------------------------

    @staticmethod
    def _foot_trajectory(phase, nominal, stride, lift, stance_ratio):
        phase = float(phase) % 1.0
        if phase < stance_ratio:
            u = phase / stance_ratio
            offset = (0.5 - u) * stride
            z_offset = 0.0
        else:
            u = (phase - stance_ratio) / (1.0 - stance_ratio)
            offset = (-0.5 + u) * stride
            z_offset = lift * np.sin(np.pi * u)

        return nominal + offset + np.array([0.0, 0.0, z_offset])

    def _update_walk_targets(self, dt):
        self.gait_phase = (
            self.gait_phase + dt / max(self.gait_period, 1e-4)
        ) % 1.0

        vx, vy, wz = self.command
        for leg in range(6):
            r = SopHexapodGeometry.LEG_ORIGINS[leg]

            # Translational stride plus the tangential component of yaw motion.
            stride_xy = np.array([
                self.max_stride_x * vx,
                self.max_stride_y * vy,
            ])
            yaw_xy = self.max_yaw_stride * wz * np.array([
                -r[1] / 0.093,
                 r[0] / 0.093,
            ])
            stride_xy = stride_xy + yaw_xy
            stride = np.array([stride_xy[0], stride_xy[1], 0.0])

            phase = (
                self.gait_phase + SopHexapodGeometry.GAIT_OFFSETS[leg]
            ) % 1.0

            foot_target = self._foot_trajectory(
                phase,
                self.nominal_feet[leg],
                stride,
                self.step_height,
                self.stance_ratio,
            )
            self.joint_targets[leg] = self._solve_leg_ik(
                leg, foot_target
            )
            # if leg == 0:
            #     print("Foot target:", foot_target)
            #     print("Joint target:", self.joint_targets[leg])

    # ------------------------------------------------------------------
    # Main control update
    # ------------------------------------------------------------------

    def update(self, dt=None):
        """Update state machine and actuator targets. Call at ~50 Hz."""
        if dt is None:
            dt = self.control_period
        dt = float(dt)

        if self._is_transitioning:
            self._transition_elapsed += dt
            alpha = min(
                1.0,
                self._transition_elapsed / max(self.transition_duration, 1e-4)
            )
            # Smoothstep interpolation: zero velocity at both endpoints.
            s = alpha * alpha * (3.0 - 2.0 * alpha)
            self.joint_targets = (
                (1.0 - s) * self._transition_start
                + s * self._transition_goal
            )
            if alpha >= 1.0:
                self._is_transitioning = False
                self.state = self._transition_final_state
                self.joint_targets = self._transition_goal.copy()

                if self._pending_state == RobotState.WALK:
                    self._pending_state = None
                    self.state = RobotState.WALK
                    self.gait_phase = 0.0

        elif self.state == RobotState.WALK:
            if np.linalg.norm(self.command) < 1e-4:
                self.request_state(RobotState.STAND)
            else:
                self._update_walk_targets(dt)

        self._write_actuators(self.joint_targets)
        # print(
        #     "Base position:",
        #     self.data.xpos[self.base_body_id]
        # )

        # print(
        #     "Base velocity:",
        #     self.data.qvel[:3]
        # )
        # print("Contacts:", self.data.ncon)
        


    def get_status(self):
        return {
            "state": self.state.name,
            "command": self.command.copy(),
            "transitioning": self._is_transitioning,
            "gait_phase": float(self.gait_phase),
        }