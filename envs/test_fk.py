import numpy as np
from SopEngine import LegKinematics, FootTrajectory
# 假设已经导入：
# LegKinematics
# FootTrajectory

# 使用你之前计算出的六条腿 nominal positions
nominal_positions = np.array([
    [-0.34624407,  0.33391646, 0.00616859],
    [ 0.00000000,  0.47023548, 0.00616859],
    [ 0.34624407,  0.33391646, 0.00616859],
    [ 0.34624407, -0.33391646, 0.00616859],
    [ 0.00000000, -0.47023548, 0.00616859],
    [-0.34624407, -0.33391646, 0.00616859],
])

stride_vector = np.array([0.04, 0.0, 0.0])

phases = np.linspace(0.0, 0.99, 100)

for leg_index in range(6):

    print(f"\nLeg {leg_index + 1}")

    initial_angles = np.array([0.0, 0.0, -0.8])

    failed_phases = []
    max_error = 0.0

    for phase in phases:

        target = FootTrajectory.generate(
            phase=phase,
            nominal_position=nominal_positions[leg_index],
            stride_vector=stride_vector
        )

        angles, success, error = LegKinematics.inverse_kinematics(
            target_position=target,
            leg_index=leg_index,
            initial_angles=initial_angles
        )

        max_error = max(max_error, error)

        if not success:
            failed_phases.append(phase)

        # 使用当前解作为下一次 IK 的初始值
        initial_angles = angles

    print("Failed phases:", failed_phases)
    print("Maximum IK error:", max_error)