import numpy as np
from enum import Enum, auto
from dataclasses import dataclass


# ============================================================
# 1. Robot Parameters
# ============================================================

@dataclass
class RobotConfig:
    # Robot structure
    num_legs: int = 6
    joints_per_leg: int = 3

    # Link dimensions (meters)
    coxa_length: float = 0.06
    femur_length: float = 0.12
    tibia_length: float = 0.22

    # Joint limits (radians)
    coxa_range: tuple = (-0.6, 0.6)
    femur_range: tuple = (-1.5, 1.5)
    tibia_range: tuple = (-1.6, -0.2)

    @property
    def num_joints(self):
        return self.num_legs * self.joints_per_leg

def quaternion_to_matrix(quat):
    """
    Convert quaternion to rotation matrix.

    Quaternion format:
        [w, x, y, z]

    This is the format used by MuJoCo.
    """
    quat = np.asarray(quat, dtype=float)

    if quat.shape != (4,):
        raise ValueError("Quaternion must have 4 elements.")

    norm = np.linalg.norm(quat)

    if norm < 1e-9:
        raise ValueError("Invalid zero-length quaternion.")

    w, x, y, z = quat / norm

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

# ============================================================
# 2. Leg Mounting Positions
# ============================================================

class HexapodGeometry:

    # Coxa joint positions relative to robot body frame
    LEG_ORIGINS = np.array([
        [-0.079498,  0.067171, 0.0],  # Leg 1
        [ 0.000000,  0.093000, 0.0],  # Leg 2
        [ 0.079498,  0.067171, 0.0],  # Leg 3
        [ 0.079498, -0.067171, 0.0],  # Leg 4
        [ 0.000000, -0.093000, 0.0],  # Leg 5
        [-0.079498, -0.067171, 0.0],  # Leg 6
    ])

    LEG_QUATERNIONS = np.array([
    [0.382683, 0, 0,  0.923880],
    [0.707107, 0, 0,  0.707107],
    [0.923880, 0, 0,  0.382683],
    [0.923880, 0, 0, -0.382683],
    [0.707107, 0, 0, -0.707107],
    [0.382683, 0, 0, -0.923880],
    ])

    # Convert mounting quaternions into rotation matrices
    LEG_ROTATIONS = np.array([
        quaternion_to_matrix(q)
        for q in LEG_QUATERNIONS
    ])

# ============================================================
# 3. Mathematical Utilities
# ============================================================

def clamp(value, lower, upper):
    """
    Restrict a value to [lower, upper].
    """
    return np.clip(value, lower, upper)


def wrap_angle(angle):
    """
    Normalize angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_vector(vector, eps=1e-9):
    """
    Normalize a 3D vector.
    """
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm < eps:
        raise ValueError("Cannot normalize a zero-length vector.")

    return vector / norm


def rotation_x(angle):
    """
    Rotation matrix around X axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_y(angle):
    """
    Rotation matrix around Y axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_z(angle):
    """
    Rotation matrix around Z axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])



def rotate_vector(vector, rotation_matrix):
    """
    Rotate a vector using a rotation matrix.
    """
    vector = np.asarray(vector, dtype=float)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)

    return rotation_matrix @ vector

# ============================================================
# 4. Coordinate Transformation
# ============================================================

class CoordinateTransformation:

    @staticmethod
    def body_to_leg(position, leg_index):
        """
        Transform a position from robot body frame
        to the specified leg frame.

        Args:
            position: Position in body frame (3,)
            leg_index: Leg index, from 0 to 5

        Returns:
            Position in leg frame (3,)
        """

        position = np.asarray(position, dtype=float)

        if not 0 <= leg_index < 6:
            raise IndexError("Invalid leg index.")

        origin = HexapodGeometry.LEG_ORIGINS[leg_index]
        rotation = HexapodGeometry.LEG_ROTATIONS[leg_index]

        return rotation.T @ (position - origin)

    @staticmethod
    def leg_to_body(position, leg_index):
        """
        Transform a position from leg frame
        to robot body frame.
        """

        position = np.asarray(position, dtype=float)

        if not 0 <= leg_index < 6:
            raise IndexError("Invalid leg index.")

        origin = HexapodGeometry.LEG_ORIGINS[leg_index]
        rotation = HexapodGeometry.LEG_ROTATIONS[leg_index]

        return rotation @ position + origin


# ============================================================
# 6. Homogeneous Transformation
# ============================================================

def make_transform(rotation, translation):
    """
    Construct a 4x4 homogeneous transformation matrix.
    """

    T = np.eye(4)

    T[:3, :3] = rotation
    T[:3, 3] = translation

    return T


def translation_matrix(position):
    """
    Construct a pure translation matrix.
    """

    return make_transform(
        np.eye(3),
        np.asarray(position, dtype=float)
    )


def rotation_matrix_transform(rotation):
    """
    Convert a 3x3 rotation matrix to a 4x4 transform.
    """

    return make_transform(
        rotation,
        np.zeros(3)
    )


def axis_angle_rotation(axis, angle):
    """
    Construct a rotation matrix from an axis and angle.
    """

    axis = normalize_vector(axis)

    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    return np.array([
        [x*x*C+c,   x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, y*y*C+c,   y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
    ])

# ============================================================
# 7. Leg Kinematics
# ============================================================

class LegKinematics:

    # Coxa joint to Femur joint
    COXA_TO_FEMUR = np.array([0.06, 0.0, 0.0])

    # Femur joint to Tibia joint
    FEMUR_TO_TIBIA = np.array([0.0, 0.0, -0.12])

    # Tibia joint to Foot Site
    TIBIA_TO_FOOT = np.array([0.0, 0.0, -0.22])

    # Fixed orientation of Femur body
    FEMUR_QUAT = np.array([
        0.5, 0.0, -0.866025, 0.0
    ])

    # Fixed orientation of Tibia body
    TIBIA_QUAT = np.array([
        0.707107, 0.0, 0.707107, 0.0
    ])

    # Joint axes in their respective local frames
    COXA_AXIS = np.array([0.0, 0.0, -1.0])
    FEMUR_AXIS = np.array([0.0, 1.0, 0.0])
    TIBIA_AXIS = np.array([0.0, 1.0, 0.0])

    @staticmethod
    def forward_kinematics(joint_angles, leg_index):
        """
        Calculate the foot position of one leg
        in the robot body frame.

        Args:
            joint_angles:
                [coxa_angle, femur_angle, tibia_angle]

            leg_index:
                Leg index from 0 to 5.

        Returns:
            Foot position in body frame (3,)
        """

        q1, q2, q3 = np.asarray(
            joint_angles,
            dtype=float
        )

        if not 0 <= leg_index < 6:
            raise IndexError("Invalid leg index.")

        # Mounting transformation
        origin = HexapodGeometry.LEG_ORIGINS[leg_index]

        R_mount = HexapodGeometry.LEG_ROTATIONS[leg_index]

        T = make_transform(R_mount, origin)

        # Coxa joint rotation
        R_coxa = axis_angle_rotation(
            LegKinematics.COXA_AXIS,
            q1
        )

        T = T @ rotation_matrix_transform(R_coxa)

        # Coxa -> Femur
        T = T @ translation_matrix(
            LegKinematics.COXA_TO_FEMUR
        )

        # Fixed Femur orientation
        R_femur_fixed = quaternion_to_matrix(
            LegKinematics.FEMUR_QUAT
        )

        T = T @ rotation_matrix_transform(
            R_femur_fixed
        )

        # Femur joint rotation
        R_femur = axis_angle_rotation(
            LegKinematics.FEMUR_AXIS,
            q2
        )

        T = T @ rotation_matrix_transform(R_femur)

        # Femur -> Tibia
        T = T @ translation_matrix(
            LegKinematics.FEMUR_TO_TIBIA
        )

        # Fixed Tibia orientation
        R_tibia_fixed = quaternion_to_matrix(
            LegKinematics.TIBIA_QUAT
        )

        T = T @ rotation_matrix_transform(
            R_tibia_fixed
        )

        # Tibia joint rotation
        R_tibia = axis_angle_rotation(
            LegKinematics.TIBIA_AXIS,
            q3
        )

        T = T @ rotation_matrix_transform(R_tibia)

        # Tibia -> Foot Site
        T = T @ translation_matrix(
            LegKinematics.TIBIA_TO_FOOT
        )

        return T[:3, 3]

    @staticmethod
    def calculate_jacobian(joint_angles, leg_index, epsilon=1e-6):
        """
        Calculate the numerical Jacobian of one leg.

        Args:
            joint_angles: Three joint angles.
            leg_index: Leg index from 0 to 5.
            epsilon: Numerical differentiation step.

        Returns:
            Jacobian matrix with shape (3, 3).
        """

        joint_angles = np.asarray(joint_angles, dtype=float)

        if joint_angles.shape != (3,):
            raise ValueError("joint_angles must have shape (3,)")

        J = np.zeros((3, 3))

        for i in range(3):

            q_plus = joint_angles.copy()
            q_minus = joint_angles.copy()

            q_plus[i] += epsilon
            q_minus[i] -= epsilon

            p_plus = LegKinematics.forward_kinematics(q_plus, leg_index)
            p_minus = LegKinematics.forward_kinematics(q_minus, leg_index)

            J[:, i] = (p_plus - p_minus) / (2 * epsilon)

        return J

    @staticmethod
    def inverse_kinematics(
        target_position,
        leg_index,
        initial_angles=None,
        max_iterations=100,
        tolerance=1e-5,
        damping=0.01
    ):
        """
        Numerical inverse kinematics using Damped Least Squares.

        Args:
            target_position:
                Target foot position in body frame.

            leg_index:
                Leg index from 0 to 5.

            initial_angles:
                Initial joint angles. If None, use zero angles.

            max_iterations:
                Maximum iteration count.

            tolerance:
                Position error threshold in meters.

            damping:
                DLS damping coefficient.

        Returns:
            joint_angles, success, error
        """

        target_position = np.asarray(
            target_position,
            dtype=float
        )

        if target_position.shape != (3,):
            raise ValueError("target_position must have shape (3,)")

        if not 0 <= leg_index < 6:
            raise IndexError("Invalid leg index.")

        if initial_angles is None:
            q = np.array([0.0, 0.0, -0.8])
        else:
            q = np.asarray(initial_angles, dtype=float).copy()

        if q.shape != (3,):
            raise ValueError("initial_angles must have shape (3,)")

        # Joint limits
        lower_limits = np.array([
            -0.6,
            -1.5,
            -1.6
        ])

        upper_limits = np.array([
            0.6,
            1.5,
            -0.2
        ])

        for iteration in range(max_iterations):

            current_position = LegKinematics.forward_kinematics(
                q,
                leg_index
            )

            error_vector = target_position - current_position

            error = np.linalg.norm(error_vector)

            if error < tolerance:
                return q, True, error

            J = LegKinematics.calculate_jacobian(q, leg_index)

            # Damped Least Squares
            A = J @ J.T + damping**2 * np.eye(3)

            delta_q = J.T @ np.linalg.solve(A, error_vector)

            # Limit the maximum change per iteration
            max_step = 0.1

            delta_norm = np.linalg.norm(delta_q)

            if delta_norm > max_step:
                delta_q *= max_step / delta_norm

            q += delta_q

            # Enforce joint limits
            q = np.clip(
                q,
                lower_limits,
                upper_limits
            )

        final_position = LegKinematics.forward_kinematics(q, leg_index)

        final_error = np.linalg.norm(
            target_position - final_position
        )

        return q, final_error < tolerance, final_error

class FootTrajectory:

    @staticmethod
    def generate(
        phase,
        nominal_position,
        stride_vector,
        step_height=0.03,
        stance_ratio=0.6
    ):
        """
        Generate foot trajectory in body frame.

        Args:
            phase:
                Normalized gait phase [0, 1).

            nominal_position:
                Neutral foot position in body frame.

            stride_vector:
                Horizontal displacement vector
                from rear extreme to front extreme.

            step_height:
                Maximum foot lifting height.

            stance_ratio:
                Fraction of the cycle spent in stance.

        Returns:
            target_position:
                Foot target position in body frame.
        """

        nominal_position = np.asarray(
            nominal_position,
            dtype=float
        )

        stride_vector = np.asarray(
            stride_vector,
            dtype=float
        )

        if nominal_position.shape != (3,):
            raise ValueError(
                "nominal_position must have shape (3,)"
            )

        if stride_vector.shape != (3,):
            raise ValueError(
                "stride_vector must have shape (3,)"
            )

        if not np.isclose(stride_vector[2], 0.0):
            raise ValueError(
                "stride_vector must be horizontal"
            )

        if step_height < 0:
            raise ValueError(
                "step_height must be non-negative"
            )

        if not 0 < stance_ratio < 1:
            raise ValueError(
                "stance_ratio must be between 0 and 1"
            )

        # Normalize phase to [0, 1)
        phase = float(phase) % 1.0

        # -------------------------
        # Stance phase
        # -------------------------
        if phase < stance_ratio:

            u = phase / stance_ratio

            # Move from front to rear
            horizontal_offset = (
                0.5 - u
            ) * stride_vector

            vertical_offset = 0.0

        # -------------------------
        # Swing phase
        # -------------------------
        else:

            u = (
                phase - stance_ratio
            ) / (1.0 - stance_ratio)

            # Move from rear to front
            horizontal_offset = (
                -0.5 + u
            ) * stride_vector

            # Sinusoidal foot lifting
            vertical_offset = (
                step_height * np.sin(np.pi * u)
            )

        target_position = (
            nominal_position
            + horizontal_offset
            + np.array([0.0, 0.0, vertical_offset])
        )

        return target_position