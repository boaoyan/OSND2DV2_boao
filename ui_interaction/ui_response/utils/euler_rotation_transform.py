import numpy as np




def rotation_to_euler_angles(rt):
    """从 4x4 齐次矩阵提取 XYZ 外旋欧拉角 [roll, pitch, yaw]"""
    assert rt.shape == (4, 4)
    R = rt[:3, :3]
    t = rt[:3, 3]

    # 正确：pitch = -arcsin(R[2, 0])
    if abs(R[2, 0]) < 1.0:
        theta_y = -np.arcsin(R[2, 0])  # pitch
        cos_theta_y = np.cos(theta_y)
        if abs(cos_theta_y) > 1e-6:
            # roll = atan2(R[2,1], R[2,2])
            theta_x = np.arctan2(R[2, 1], R[2, 2])
            # yaw  = atan2(R[1,0], R[0,0])
            theta_z = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock: cos(pitch) ≈ 0
            theta_x = 0.0
            theta_z = np.arctan2(-R[0, 1], R[1, 1])
    else:
        # |R[2,0]| == 1 → gimbal lock
        theta_y = np.pi / 2 if R[2, 0] <= -1.0 else -np.pi / 2
        theta_x = 0.0
        theta_z = np.arctan2(R[0, 1], R[0, 2])

    return np.array([[theta_x, theta_y, theta_z, t[0], t[1], t[2]]])


def euler_angles_to_rotation(euler):
    """
    将 [theta_x, theta_y, theta_z, tx, ty, tz] 转换为 4x4 齐次变换矩阵。
    假设旋转顺序为 XYZ 外旋（等价于 ZYX 内旋），与 rotation_to_euler_angles 互逆。

    Parameters:
        euler: array-like, shape (6,) or (1,6)
            [roll (x), pitch (y), yaw (z), tx, ty, tz]

    Returns:
        rt: np.ndarray, shape (4, 4)
    """
    euler = np.asarray(euler)
    if euler.ndim == 2:
        if euler.shape[0] == 1 and euler.shape[1] == 6:
            euler = euler[0]
        else:
            raise ValueError("Expected shape (6,) or (1,6), got {}".format(euler.shape))
    elif euler.ndim != 1 or euler.size != 6:
        raise ValueError("Input must be a 6-element vector")

    theta_x, theta_y, theta_z, tx, ty, tz = euler

    # 绕 X 轴旋转（Roll）
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    # 绕 Y 轴旋转（Pitch）
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    # 绕 Z 轴旋转（Yaw）
    cz = np.cos(theta_z)
    sz = np.sin(theta_z)
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    # 注意：外旋 XYZ 等价于先绕固定坐标系 X，再 Y，再 Z
    # 所以总旋转矩阵为：R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    # 构造齐次变换矩阵
    rt = np.eye(4)
    rt[:3, :3] = R
    rt[:3, 3] = [tx, ty, tz]

    return rt





def reg_rt_update_per_dof(euler1, euler2, var1, var2):
    """
    每个自由度独立融合（6DoF）

    Parameters:
        euler1, euler2: array-like, shape (6,) or (1,6)
            [roll, pitch, yaw, tx, ty, tz]
        var1, var2: array-like, shape (6,)
            对应每个自由度的方差（必须为正）

    Returns:
        fused_euler: np.ndarray, shape (6,)
    """
    # 统一转为 1D
    e1 = np.atleast_1d(np.squeeze(euler1))
    e2 = np.atleast_1d(np.squeeze(euler2))
    var1 = np.atleast_1d(np.squeeze(var1))
    var2 = np.atleast_1d(np.squeeze(var2))

    assert e1.shape == e2.shape == (6,), f"Expected (6,), got {e1.shape}, {e2.shape}"
    assert var1.shape == var2.shape == (6,), f"Var shape mismatch: {var1.shape}, {var2.shape}"

    # 计算每个自由度的权重: w1_i = var2_i / (var1_i + var2_i)
    # 注意：避免除零
    total_var = var1 + var2
    total_var = np.where(total_var == 0, 1e-12, total_var)  # 防止除零
    w1 = var2 / total_var  # weight for euler1
    w2 = var1 / total_var  # weight for euler2 (w2 = 1 - w1)

    # --- 处理旋转部分 (roll, pitch, yaw) ---
    r1 = e1[:3]
    r2 = e2[:3]
    r_fused = np.empty(3)
    for i in range(3):
        # 对每个角度做圆周加权平均
        x = w1[i] * np.cos(r1[i]) + w2[i] * np.cos(r2[i])
        y = w1[i] * np.sin(r1[i]) + w2[i] * np.sin(r2[i])
        r_fused[i] = np.arctan2(y, x)

    # --- 处理平移部分 (tx, ty, tz) ---
    t1 = e1[3:]
    t2 = e2[3:]
    t_fused = w1[3:] * t1 + w2[3:] * t2  # 线性加权

    # 合并
    fused = np.concatenate([r_fused, t_fused])
    return fused


# def euler_angles_to_rotation(R):
#     """
#     X-Y-Z 主动外旋旋转矩阵
#     R = Rx(theta_x) * Ry(theta_y) * Rz(theta_z)
#     """
#     R = np.asarray(R)
#     if R.ndim == 2 and R.shape[0] == 1:
#         R = R[0]  # 降维到 (6,)
#     elif R.ndim != 1 or R.size != 6:
#         raise ValueError(f"Expected 6-element 1D array, got shape {R.shape}")
#
#     Tx, Ty, Tz, A, B, C = R
#     sA = np.sin(A)
#     cA = np.cos(A)
#     sB = np.sin(B)
#     cB = np.cos(B)
#     sC = np.sin(C)
#     cC = np.cos(C)
#
#     R = np.array([
#         [cB * cC, -cB * sC, sB],
#         [cA * sC + sA * sB * cC, cA * cC - sA * sB * sC, -cB * sA],
#         [sA * sC - cA * sB * cC, sA * cC + cA * sB * sC, cA * cB]
#     ])
#
#     T = np.array([[Tx], [Ty], [Tz]])
#
#     RT = np.eye(4)
#     RT[:3, :3] = R
#     RT[:3, 3:4] = T
#
#     return  RT
