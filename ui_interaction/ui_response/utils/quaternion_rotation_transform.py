import numpy as np


def quaternion_to_rt_matrix(q):
    """
    将7维位姿向量 [w, x, y, z, tx, ty, tz] 转换为 4x4 齐次变换矩阵。

    参数:
        q (array-like): 7维向量，前4维为四元数 [w, x, y, z]，后3维为平移 [tx, ty, tz]

    返回:
        T (np.ndarray): 4x4 齐次变换矩阵，形式为:
            [[R00, R01, R02, tx],
             [R10, R11, R12, ty],
             [R20, R21, R22, tz],
             [  0,   0,   0,  1]]
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.size != 7:
        raise ValueError("Input must be a 7-element vector [w, x, y, z, tx, ty, tz].")

    # 拆分四元数和平移分量
    quat = q[3:7]
    trans = q[:3]

    # 归一化四元数
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat

    # 构建3x3旋转矩阵（逐元素赋值，保持数值稳定性）
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    R[0, 1] = 2 * x * y - 2 * w * z
    R[0, 2] = 2 * x * z + 2 * w * y

    R[1, 0] = 2 * x * y + 2 * w * z
    R[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    R[1, 2] = 2 * y * z - 2 * w * x

    R[2, 0] = 2 * x * z - 2 * w * y
    R[2, 1] = 2 * y * z + 2 * w * x
    R[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2

    # 构建4x4齐次变换矩阵
    RT = np.eye(4, dtype=np.float64)
    RT[:3, :3] = R          # 旋转部分
    RT[:3, 3] = trans       # 平移部分

    return RT





def rt_matrix_to_quaternion_pose(T):
    """
    将 4x4 齐次变换矩阵转换为 7 维位姿向量 [w, x, y, z, tx, ty, tz]，
    其中前4维为单位四元数，后3维为平移向量。

    参数:
        T (array-like): 4x4 齐次变换矩阵，格式为:
            [[R00, R01, R02, tx],
             [R10, R11, R12, ty],
             [R20, R21, R22, tz],
             [  0,   0,   0,  1]]

    返回:
        pose (np.ndarray): 7维向量 [w, x, y, z, tx, ty, tz]
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Input must be a 4x4 homogeneous transformation matrix.")

    # 提取旋转矩阵 (3x3) 和平移向量 (3,)
    R = T[:3, :3]
    trans = T[:3, 3]

    # === 旋转矩阵 → 四元数（数值稳定实现）===
    # 提取矩阵元素
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # 计算各分量平方（带防负处理）
    q0_sq = max(1.0 + r11 + r22 + r33, 0.0)
    q1_sq = max(1.0 + r11 - r22 - r33, 0.0)
    q2_sq = max(1.0 - r11 + r22 - r33, 0.0)
    q3_sq = max(1.0 - r11 - r22 + r33, 0.0)

    q0 = np.sqrt(q0_sq) / 2.0
    q1 = np.sqrt(q1_sq) / 2.0
    q2 = np.sqrt(q2_sq) / 2.0
    q3 = np.sqrt(q3_sq) / 2.0

    # 基于最大分量选择计算路径（避免除零）
    q_vals = [q0, q1, q2, q3]
    idx = int(np.argmax(q_vals))

    if idx == 0:  # q0 (w) 最大
        q0 = q0
        q1 = (r32 - r23) / (4.0 * q0 + 1e-12)  # 防除零
        q2 = (r13 - r31) / (4.0 * q0 + 1e-12)
        q3 = (r21 - r12) / (4.0 * q0 + 1e-12)
    elif idx == 1:  # q1 (x) 最大
        q1 = q1
        q0 = (r32 - r23) / (4.0 * q1 + 1e-12)
        q2 = (r12 + r21) / (4.0 * q1 + 1e-12)
        q3 = (r13 + r31) / (4.0 * q1 + 1e-12)
    elif idx == 2:  # q2 (y) 最大
        q2 = q2
        q0 = (r13 - r31) / (4.0 * q2 + 1e-12)
        q1 = (r12 + r21) / (4.0 * q2 + 1e-12)
        q3 = (r23 + r32) / (4.0 * q2 + 1e-12)
    else:  # idx == 3, q3 (z) 最大
        q3 = q3
        q0 = (r21 - r12) / (4.0 * q3 + 1e-12)
        q1 = (r31 + r13) / (4.0 * q3 + 1e-12)
        q2 = (r32 + r23) / (4.0 * q3 + 1e-12)

    quat = np.array([q0, q1, q2, q3], dtype=np.float64)
    quat /= np.linalg.norm(quat)  # 归一化为单位四元数

    # 拼接为7维位姿向量 [tx, ty, tz, w, x, y, z]
    pose = np.concatenate([trans, quat])

    return pose