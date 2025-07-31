import numpy as np
from scipy.linalg import svd


def compute_4x4_transform(p, q):
    """
    最佳旋转算法；p和q应该是尺寸 (N，3) 的三维点集合，其中N是点数。返回p2q的旋转平移变换
    :param p: 三维点集p
    :param q: 三维点集q
    :return: 返回一个旋转矩阵，平移向量，使得 p = (R_matrix @ q.T + Trans_vec.reshape((3, 1))).T，从q的坐标系转换到p的坐标系
    """

    # 计算两个点集的质心。
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    # 通过减去它们的质心来使点集居中。
    p_centered = p - centroid_p
    q_centered = q - centroid_q

    # 计算中心点集的协方差矩阵。
    cov = p_centered.T.dot(q_centered)

    # 计算协方差矩阵的奇异值分解。
    U, S, V = np.linalg.svd(cov)

    # 通过取U和V矩阵的点积来计算旋转矩阵。
    r_matrix = U.dot(V)

    # 通过取质心的差异来计算平移矢量
    # 两个点集，并将其乘以旋转矩阵。
    trans_vec = centroid_p - r_matrix.dot(centroid_q)

    # 最后，将旋转矩阵和平移矢量堆叠成一个4x4变换矩阵。
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = r_matrix
    transform_matrix[:3, 3] = trans_vec.flatten()

    return np.linalg.inv(transform_matrix)


def kabsch(P, Q, m=None):
    """
    使用 Kabsch 算法计算两组配对点集之间的最优刚体变换（旋转和平移），
    并返回最小化的加权 RMSD。

    参考:
    1) Kabsch W. (1976). A solution for the best rotation to relate two sets of vectors.
    2) https://en.wikipedia.org/wiki/Kabsch_algorithm

    参数:
        P : (D, N) 数组 - 第一组点，每列是一个 D 维点
        Q : (D, N) 数组 - 第二组点（与 P 配对）
        m : (N,) 数组 - 可选权重（非负），默认等权重

    返回:
        U : (D, D) 数组 - 最优旋转矩阵（正交，det=1）
        r : (D, 1) 数组 - 平移向量
        lrms : float - 加权最小均方根距离 (Least RMS)
    """
    P = np.array(P)
    Q = np.array(Q)

    D, N = P.shape

    if P.shape != Q.shape:
        raise ValueError("P 和 Q 必须具有相同的形状 (D, N)")

    if len(P.shape) != 2:
        raise ValueError("P 和 Q 必须是二维矩阵")

    # 处理权重 m
    if m is not None:
        m = np.array(m)
        if m.shape != (N,) and m.shape != (1, N):
            raise ValueError("权重 m 必须是长度为 N 的行向量或一维数组")
        m = m.flatten()
        if np.any(m < 0):
            raise ValueError("权重必须非负")
        if np.sum(m) == 0:
            raise ValueError("权重中至少有一个正数")
        m = m / np.sum(m)  # 归一化，使总和为 1
    else:
        m = np.ones(N) / N  # 等权重

    # 计算质心
    p0 = P @ m  # (D,) - 加权平均: sum_i m_i * P_i
    q0 = Q @ m  # (D,)

    # 将点集中心化（减去质心）
    P_centered = P - np.outer(p0, np.ones(N))  # P - p0 @ [1,1,...,1]
    Q_centered = Q - np.outer(q0, np.ones(N))  # Q - q0 @ [1,1,...,1]

    # 计算协方差矩阵 C = P_centered @ diag(m) @ Q_centered.T
    # 高效方式：避免显式构造 diag(m)
    P_weighted = P_centered * m  # (D, N): 每列乘以对应权重 m_i
    C = P_weighted @ Q_centered.T  # (D, D)

    # SVD 分解
    V, S, Wt = np.linalg.svd(C)
    W = Wt.T  # 因为 svd 返回 V, S, Wt 满足 C = V @ diag(S) @ Wt

    # 判断是否需要反射修正（确保 det(U) = +1，即纯旋转）
    d = np.sign(np.linalg.det(W @ V.T))
    if d < 0:
        S[-1] *= -1
        # 构造 I 矩阵，仅最后一项为 -1
        I = np.eye(D)
        I[-1, -1] = -1
        U = W @ I @ V.T
    else:
        U = W @ V.T  # 标准情况

    # 平移向量: r = q0 - U @ p0
    r = q0[:, np.newaxis] - U @ p0[:, np.newaxis]  # 列向量 (D, 1)

    # 计算最小 RMSD
    # 差值: U @ P_centered - Q_centered
    diff = U @ P_centered - Q_centered  # (D, N)
    # 加权平方误差和: sum_i m_i * ||diff_i||^2
    weighted_sq_error = np.sum((diff * diff) @ m)  # scalar
    lrms = np.sqrt(weighted_sq_error)

    return U, r, lrms

def find_3d_affine_transform(in_points_Reorder, out_points):
    """
    计算两个三维点集之间的仿射变换。

    参数:
    in_points (numpy.ndarray): 输入的三维点集，形状为 (3, N)。
    out_points (numpy.ndarray): 输出的三维点集，形状为 (3, N)。

    返回:
    numpy.ndarray: 仿射变换矩阵，形状为 (4, 4)。
    """

    # 检查输入的两个矩阵的列数是否相同
    if in_points_Reorder.shape[1] != out_points.shape[1]:
        raise ValueError("Find3DAffineTransform(): input data mis-match")

    # 计算输入和输出点集之间的比例因子
    # dist_in = np.sum(np.linalg.norm(in_points[:, 1:] - in_points[:, :-1], axis=0))
    # dist_out = np.sum(np.linalg.norm(out_points[:, 1:] - out_points[:, :-1], axis=0))
    # if dist_in <= 0 or dist_out <= 0:
    #     return np.eye(4)
    #
    # scale = dist_out / dist_in
    # out_points /= scale

    # 计算输入和输出点集的中心点
    in_ctr = np.mean(in_points_Reorder.T, axis=1)
    out_ctr = np.mean(out_points.T, axis=1)

    print("提供的坐标的中心点：\n", in_ctr)
    print("模板坐标的中心点：\n", out_ctr)
    # 将点集平移到原点
    in_points_centered = in_points_Reorder.T - in_ctr.reshape(3, 1)
    out_points_centered = out_points.T - out_ctr.reshape(3, 1)

    # 计算协方差矩阵并进行SVD分解
    cov_matrix = in_points_centered @ out_points_centered.T
    U, s, Vt = svd(cov_matrix)

    # 计算旋转矩阵
    d = np.linalg.det(Vt @ U.T)
    I = np.eye(3)
    I[2, 2] = d
    R = Vt @ I @ U.T

    # 计算最终的仿射变换矩阵
    # T = scale * (out_ctr - R @ in_ctr)
    # transform_matrix = np.eye(4)
    # transform_matrix[:3, :3] = scale * R
    # transform_matrix[:3, 3] = T

    # 计算平移向量（注意这里没有使用缩放因子）
    T = out_ctr - R @ in_ctr

    # 构建最终的仿射变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T

    return transform_matrix

import numpy as np


def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    mid_P = R @ P.T
    mid_Q = Q.T
    t = np.mean(mid_Q - mid_P, axis=1)
    return R, t, rmsd