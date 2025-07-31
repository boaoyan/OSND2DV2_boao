import numpy as np
from scipy.optimize import minimize


def rotation_matrix_to_euler_angles(R):
    # 确保R是一个3x3的旋转矩阵
    assert R.shape == (3, 3)

    # 提取各个元素
    r31 = R[2, 0]
    r11 = R[0, 0]
    r21 = R[1, 0]
    r32 = R[2, 1]
    r33 = R[2, 2]

    # 计算欧拉角
    theta_x = np.arctan2(r32, r33)  # Roll
    theta_y = np.arctan2(-r31, np.sqrt(r11 ** 2 + r21 ** 2))  # Pitch
    theta_z = np.arctan2(r21, r11)  # Yaw
    return theta_x, theta_y, theta_z


def homogeneous_to_euler_and_translation(T, sequence='zyx'):
    """
    将 4x4 齐次变换矩阵分解为平移 (Tx, Ty, Tz) 和欧拉角 (A, B, C)

    参数:
        T: 4x4 numpy 数组，齐次变换矩阵
        sequence: 旋转顺序，如 'zyx'（默认，常用）

    返回:
        Tx, Ty, Tz, A, B, C（单位：弧度）
    """
    # 提取平移部分
    Tx, Ty, Tz = T[0, 3], T[1, 3], T[2, 3]

    # 提取旋转矩阵 (3x3)
    R = T[0:3, 0:3]

    A, B, C = rotation_matrix_to_euler_angles(R)
    return Tx, Ty, Tz, A, B, C


def goto(param_toolP, param_cz, param_R2C, P0):
    [X1, Y1, Z1] = param_toolP[0]
    [X2, Y2, Z2] = param_toolP[1]
    c, z, = param_cz
    param_R2C = homogeneous_to_euler_and_translation(param_R2C)
    Tx, Ty, Tz, A, B, C = param_R2C
    # print("1:", Tx, Ty, Tz, X1, Y1, Z1, X2, Y2, Z2, z, c)
    sc = np.sin(c)
    cc = np.cos(c)
    sA = np.sin(A)
    cA = np.cos(A)
    sB = np.sin(B)
    cB = np.cos(B)
    sC = np.sin(C)
    cC = np.cos(C)

    def P(a, b, X1, Y1, Z1):
        sa = np.sin(a)
        ca = np.cos(a)
        sb = np.sin(b)
        cb = np.cos(b)
        # print("1:", Tx, Ty, Tz, X1, Y1, Z1, X2, Y2, Z2, z, c)
        x = X1 * (cb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) + sb * (sB * ca - cB * sC * sa)) - Tx - Z1 * (
                sb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) - cb * (sB * ca - cB * sC * sa)) - Y1 * (
                    cc * (sB * sa + cB * sC * ca) - cB * cC * sc) - z * (sB * ca - cB * sC * sa),
        y = Y1 * (sc * (cA * sC + cC * sA * sB) + cc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) - z * (
                sa * (cA * cC - sA * sB * sC) - cB * sA * ca) - Ty + X1 * (cb * (
                cc * (cA * sC + cC * sA * sB) - sc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) + sb * (
                                                                                   sa * (
                                                                                   cA * cC - sA * sB * sC) - cB * sA * ca)) - Z1 * (
                    sb * (cc * (cA * sC + cC * sA * sB) - sc * (
                    ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) - cb * (
                            sa * (cA * cC - sA * sB * sC) - cB * sA * ca)),
        Z = Y1 * (cc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa) + sc * (sA * sC - cA * cC * sB)) - z * (
                sa * (cC * sA + cA * sB * sC) + cA * cB * ca) - Tz + X1 * (
                    sb * (sa * (cC * sA + cA * sB * sC) + cA * cB * ca) + cb * (
                    cc * (sA * sC - cA * cC * sB) - sc * (
                    ca * (cC * sA + cA * sB * sC) - cA * cB * sa))) - Z1 * (sb * (
                cc * (sA * sC - cA * cC * sB) - sc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa)) - cb * (
                                                                                    sa * (
                                                                                    cC * sA + cA * sB * sC) + cA * cB * ca))
        # print(x[0],y[0],Z)
        return np.array([x[0], y[0], Z])

    # 目标函数
    def objective_function(params):
        a, b = params
        # 计算两点的坐标
        P1 = np.squeeze(P(a, b, X1, Y1, Z1))  # 确保是 1D 数组
        P2 = np.squeeze(P(a, b, X2, Y2, Z2))  # 确保是 1D 数组

        # 计算连线与目标点的距离
        line_vec = P2 - P1
        point_vec = P1 - P0
        cross_prod = np.cross(line_vec, point_vec)

        distance = np.linalg.norm(cross_prod) / np.linalg.norm(line_vec)

        return distance

    # 初始猜测
    initial_guess = [0.5, 0.5]

    # 优化边界
    bounds = [(0, (2200 / 2048) * np.pi), (0, (1000 / 2048) * np.pi)]  # 角度范围和插值范围

    # 优化求解
    result = minimize(objective_function, initial_guess, bounds=bounds)

    # 输出结果
    if result.success:
        a_sol, b_sol = result.x
        print(f"优化成功: a={a_sol}, b={b_sol}")
    else:
        a_sol, b_sol = result.x
        print("优化失败:", result.message)

    return a_sol, b_sol  # a,b角度是弧度制


def goto_test(param_toolP, param_cz, param_R2C, P0):
    [X1, Y1, Z1] = param_toolP[0]
    [X2, Y2, Z2] = param_toolP[1]
    c, z, = param_cz
    Tx, Ty, Tz, A, B, C = param_R2C
    # print("1:", Tx, Ty, Tz, X1, Y1, Z1, X2, Y2, Z2, z, c)
    sc = np.sin(c)
    cc = np.cos(c)
    sA = np.sin(A)
    cA = np.cos(A)
    sB = np.sin(B)
    cB = np.cos(B)
    sC = np.sin(C)
    cC = np.cos(C)

    def P(a, b, X1, Y1, Z1):
        sa = np.sin(a)
        ca = np.cos(a)
        sb = np.sin(b)
        cb = np.cos(b)
        # print("1:", Tx, Ty, Tz, X1, Y1, Z1, X2, Y2, Z2, z, c)
        x = X1 * (cb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) + sb * (sB * ca - cB * sC * sa)) - Tx - Z1 * (
                sb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) - cb * (sB * ca - cB * sC * sa)) - Y1 * (
                    cc * (sB * sa + cB * sC * ca) - cB * cC * sc) - z * (sB * ca - cB * sC * sa),
        y = Y1 * (sc * (cA * sC + cC * sA * sB) + cc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) - z * (
                sa * (cA * cC - sA * sB * sC) - cB * sA * ca) - Ty + X1 * (cb * (
                cc * (cA * sC + cC * sA * sB) - sc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) + sb * (
                                                                                   sa * (
                                                                                   cA * cC - sA * sB * sC) - cB * sA * ca)) - Z1 * (
                    sb * (cc * (cA * sC + cC * sA * sB) - sc * (
                    ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) - cb * (
                            sa * (cA * cC - sA * sB * sC) - cB * sA * ca)),
        Z = Y1 * (cc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa) + sc * (sA * sC - cA * cC * sB)) - z * (
                sa * (cC * sA + cA * sB * sC) + cA * cB * ca) - Tz + X1 * (
                    sb * (sa * (cC * sA + cA * sB * sC) + cA * cB * ca) + cb * (
                    cc * (sA * sC - cA * cC * sB) - sc * (
                    ca * (cC * sA + cA * sB * sC) - cA * cB * sa))) - Z1 * (sb * (
                cc * (sA * sC - cA * cC * sB) - sc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa)) - cb * (
                                                                                    sa * (
                                                                                    cC * sA + cA * sB * sC) + cA * cB * ca))
        # print(x[0],y[0],Z)
        return np.array([x[0], y[0], Z])

    # 目标函数
    def objective_function(params):
        a, b = params
        # 计算两点的坐标
        P1 = np.squeeze(P(a, b, X1, Y1, Z1))  # 确保是 1D 数组
        P2 = np.squeeze(P(a, b, X2, Y2, Z2))  # 确保是 1D 数组

        # 计算连线与目标点的距离
        line_vec = P2 - P1
        point_vec = P1 - P0
        cross_prod = np.cross(line_vec, point_vec)

        distance = np.linalg.norm(cross_prod) / np.linalg.norm(line_vec)

        return distance

    # 初始猜测
    initial_guess = [0.5, 0.5]

    # 优化边界
    bounds = [((-1000 / 2048) * np.pi, (1000 / 2048) * np.pi),
              ((-400 / 2048) * np.pi, (500 / 2048) * np.pi)]  # 角度范围和插值范围
    # 优化求解
    result = minimize(objective_function, initial_guess, bounds=bounds)

    # 输出结果
    if result.success:
        a_sol, b_sol = result.x
        print(f"优化成功: a={a_sol}, b={b_sol}")
    else:
        a_sol, b_sol = result.x
        print("优化失败:", result.message)

    return a_sol, b_sol  # a,b角度是弧度制
