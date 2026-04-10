import numpy as np


def skew_symmetric(omega):
    """
    返回三维向量 omega 的反对称矩阵（叉乘矩阵）。
    """
    if len(omega) != 3:
        raise ValueError("Input must be a 3D vector.")
    return np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0]
    ], dtype=np.float64)


def quaternion_multiply(q1, q2):
    """
    四元数乘法：q_out = q1 * q2
    输入：q1, q2 —— [w, x, y, z]（标量在前）
    输出：q_out —— [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float64)


def ESKF_1step_RW(state, P, measures, sigma_Q, sigma_R):
    """
    ESKF 单步更新（误差状态卡尔曼滤波 + 随机游走模型）

    参数:
        state (7,)       : 当前名义状态 [px, py, pz, qw, qx, qy, qz]
        P (6,6)          : 误差状态协方差（6D：3位置 + 3小旋转）
        measures (7,)    : 当前观测 [mx, my, mz, qw_m, qx_m, qy_m, qz_m]
        sigma_Q (2,)     : 过程噪声标准差 [σ_pos, σ_rot]
        sigma_R (2,)     : 观测噪声标准差 [σ_pos_meas, σ_rot_meas]

    返回:
        state_filted (7,): 更新后的名义状态
        P_filted (6,6)   : 更新并重置后的协方差
    """
    # --- 1. 构建噪声协方差矩阵 ---
    sigma_pos_q, sigma_rot_q = sigma_Q
    sigma_pos_r, sigma_rot_r = sigma_R

    Q = np.diag(np.concatenate([
        [sigma_pos_q ** 2] * 3,
        [sigma_rot_q ** 2] * 3
    ]))

    R = np.diag(np.concatenate([
        [sigma_pos_r ** 2] * 3,
        [sigma_rot_r ** 2] * 3
    ]))

    # --- 2. 预测步骤（Random Walk：名义状态不变）---
    pos_pred = state[0:3].copy()
    quat_pred = state[3:7].copy()

    # 误差状态协方差预测
    P_pred = P + Q

    # --- 3. 更新步骤 ---
    pos_measure = measures[0:3]
    quat_measure = measures[3:7]

    # 计算预测姿态的逆（四元数共轭，因已归一化）
    quat_pred_inv = np.array([quat_pred[0], -quat_pred[1], -quat_pred[2], -quat_pred[3]])

    # 计算相对误差四元数：Δq = q_pred⁻¹ ⊗ q_meas
    quat_measure_delta = quaternion_multiply(quat_pred_inv, quat_measure)

    # 保证实部非负（最短路径）
    if quat_measure_delta[0] < 0:
        quat_measure_delta = -quat_measure_delta

    # 构造观测残差 y ∈ ℝ⁶
    # 位置残差 + 小角度近似（2 * vector_part）
    y = np.concatenate([
        pos_measure - pos_pred,
        2.0 * quat_measure_delta[1:4]  # [x, y, z] part
    ])

    # --- 4. 卡尔曼增益与误差状态更新 ---
    # 注意：此处 P_pred 和 R 都是 6x6，可直接求解 K
    # 使用 (P_pred + R) \ P_pred 更稳定，但原代码用 P_pred / (P_pred + R)
    # 由于是对角阵，等价于逐元素除法
    K = P_pred @ np.linalg.inv(P_pred + R)

    delta_state = K @ y

    # 协方差更新（Joseph form 更稳定，但此处按原式）
    I = np.eye(6)
    P_updated = (I - K) @ P_pred @ (I - K).T + K @ R @ K.T

    # --- 5. 修正名义状态 ---
    pos = pos_pred + delta_state[0:3]
    delta_theta = delta_state[3:6]  # 小旋转误差（旋转向量）

    # 将小旋转向量转换为四元数（一阶近似）
    delta_quat = np.array([1.0, delta_theta[0] / 2, delta_theta[1] / 2, delta_theta[2] / 2])
    delta_quat = delta_quat / np.linalg.norm(delta_quat)

    # 更新姿态：q = q_pred ⊗ Δq
    quat = quaternion_multiply(quat_pred, delta_quat)

    # --- 6. 重置协方差（Error Reset）---
    G = np.eye(6)
    G[3:6, 3:6] = np.eye(3) - 0.5 * skew_symmetric(delta_theta)

    P_filted = G @ P_updated @ G.T

    state_filted = np.concatenate([pos, quat])

    return state_filted, P_filted