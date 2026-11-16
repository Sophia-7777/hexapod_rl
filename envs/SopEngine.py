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


def rotate_vector(vector, rotation_matrix):
    """
    Rotate a vector using a rotation matrix.
    """
    vector = np.asarray(vector, dtype=float)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)

    return rotation_matrix @ vector

