import matplotlib.pyplot as plt
import numpy as np


def transform_points(rt_matrix, points):
    """
    将多个三维点（n, 3）与RT矩阵相乘，返回变换后的三维点。

    参数:
        rt_matrix (4x4 numpy array): 齐次变换矩阵（旋转+平移）
        points (numpy array, shape=(n, 3)): 输入的n个三维点，每行是一个 [x, y, z]

    返回:
        numpy array, shape=(n, 3): 变换后的n个三维点
    """
    if points.shape[1] != 3:
        raise ValueError("points 必须是 (n, 3) 形状的数组")

    # 转换为齐次坐标：添加一列 1 → (n, 4)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])  # shape: (n, 4)

    # 矩阵乘法: RT (4x4) @ points.T (4xn) → (4xn)，再转置为 (n, 4)
    # 更高效的方式：使用 @ 运算符和转置
    transformed_homogeneous = (rt_matrix @ points_homogeneous.T).T  # shape: (n, 4)

    # 转换回三维坐标：取前3列（x, y, z），忽略 w 分量
    # 对于仿射变换，w ≈ 1，无需除法
    transformed_points = transformed_homogeneous[:, :3]  # shape: (n, 3)

    return transformed_points


def display_cali_result(arm_pts, cam_pts, rt_arm2cam):
    # 计算变换后的机械臂点
    transformed_arm_pts = transform_points(rt_arm2cam, arm_pts)
    # 计算误差
    diff = transformed_arm_pts - cam_pts  # 残差
    mse = np.mean(np.sum(diff ** 2, axis=1))  # 均方误差（每个点距离平方的平均）
    rmse = np.sqrt(mse)  # 均方根误差

    print(f"配准 RMSE: {rmse:.4f} mm")
    # 可选：逐点误差
    point_errors = np.linalg.norm(diff, axis=1)
    print("各点误差 (mm):", np.round(point_errors, 4))

    # === 可视化部分 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 转换为 numpy 数组（确保）
    cam_pts = np.array(cam_pts)
    transformed_arm_pts = np.array(transformed_arm_pts)

    # 绘制相机检测点（真实观测）
    ax.scatter(cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2],
               color='red', label='Camera Points (Observed)', s=60, alpha=0.9)

    # 绘制变换后的机械臂点（预测）
    ax.scatter(transformed_arm_pts[:, 0], transformed_arm_pts[:, 1], transformed_arm_pts[:, 2],
               color='blue', label='Transformed Arm Points', s=60, alpha=0.9)

    # 可选：绘制从 camera points 指向 transformed points 的误差向量
    for i in range(len(cam_pts)):
        ax.plot([cam_pts[i, 0], transformed_arm_pts[i, 0]],
                [cam_pts[i, 1], transformed_arm_pts[i, 1]],
                [cam_pts[i, 2], transformed_arm_pts[i, 2]],
                color='gray', linestyle='--', linewidth=1)

    # 设置标签和标题
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Hand-Eye Calibration Result\nRMSE = {rmse:.4f} mm")

    # 添加图例
    ax.legend()

    # 网格
    ax.grid(True)

    # 自动调整视角
    plt.tight_layout()
    plt.show()
