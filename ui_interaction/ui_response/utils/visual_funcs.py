import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_line_and_point(P1, P2, aim_in_cam, extend_ratio=2.0, show_perpendicular=True,
                        title="3D Line and Target Point"):
    """
    绘制由 P1, P2 定义的直线（含延长线）和目标点 aim_in_cam

    参数：
    - P1, P2: (3,) 数组，线段的两个端点（在相机坐标系）
    - aim_in_cam: (3,) 数组，目标点
    - extend_ratio: float，延长线长度倍数（基于原线段长度）
    - show_perpendicular: bool，是否绘制从目标点到直线的垂线
    - title: str，图像标题
    """
    # 确保输入为 numpy 数组
    P1 = np.array(P1)
    P2 = np.array(P2)
    aim_in_cam = np.array(aim_in_cam)

    # 计算方向向量
    direction = P2 - P1
    length = np.linalg.norm(direction)
    if length < 1e-10:
        print("P1 和 P2 几乎重合，无法定义直线。")
        return

    unit_dir = direction / length

    # 延长线：向前和向后延伸原长度的 extend_ratio 倍
    extension = extend_ratio * length
    extended_start = P1 - unit_dir * extension  # 反向延长
    extended_end = P2 + unit_dir * extension  # 正向延长

    # 生成直线上的点用于绘图
    t = np.linspace(-extension, length + extension, 100)
    line_points = P1 + np.outer(t, unit_dir)

    # 创建 3D 图像
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制无限长直线（用细虚线）
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
            color='gray', linestyle='--', linewidth=1, label='Extended line')

    # 强调原始线段 P1-P2（实线）
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]],
            color='blue', linewidth=3, label='Segment P1-P2')

    # 绘制 P1, P2 两个点
    ax.scatter([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]],
               color='blue', s=60, depthshade=False)

    # 绘制目标点
    ax.scatter(*aim_in_cam, color='red', s=80, label='Target Point (aim)', depthshade=False)

    # 可选：绘制从目标点到直线的最短距离垂线
    if show_perpendicular:
        # 计算最近点（投影）
        point_vec = aim_in_cam - P1
        t_proj = np.dot(point_vec, unit_dir)
        closest_point = P1 + t_proj * unit_dir

        # 绘制垂线
        ax.plot([aim_in_cam[0], closest_point[0]],
                [aim_in_cam[1], closest_point[1]],
                [aim_in_cam[2], closest_point[2]],
                color='green', linestyle='-', linewidth=2, label='Shortest distance')

        # 可选：标记垂足
        ax.scatter(*closest_point, color='green', s=40, depthshade=False)

    # 设置标签和图例
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)

    # 保持比例一致（可选）
    max_range = np.array([
        line_points[:, 0].max() - line_points[:, 0].min(),
        line_points[:, 1].max() - line_points[:, 1].min(),
        line_points[:, 2].max() - line_points[:, 2].min()
    ]).max() / 2.0

    mid_x = line_points[:, 0].mean()
    mid_y = line_points[:, 1].mean()
    mid_z = line_points[:, 2].mean()

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def plot_5_points(P0, P1, aim_in_cam, pj_pt, aim_in_cam1, title="5-Point Geometry Visualization"):
    """
    可视化5个点的空间关系：
    - P0, P1: 定义直线
    - aim_in_cam: 原始目标点
    - pj_pt: 到直线的投影（垂足）
    - aim_in_cam1: 关于垂足的对称点
    """
    # 转为 numpy 数组
    P0 = np.array(P0)
    P1 = np.array(P1)
    aim_in_cam = np.array(aim_in_cam)
    pj_pt = np.array(pj_pt)
    aim_in_cam1 = np.array(aim_in_cam1)

    # 计算方向向量（用于延长线）
    direction = P1 - P0
    length = np.linalg.norm(direction)
    if length < 1e-10:
        print("P0 和 P1 几乎重合！")
        return

    unit_dir = direction / length

    # 延长线范围（前后各延伸 2 倍原长度）
    extension = 2.0 * length
    t = np.linspace(-extension, length + extension, 100)
    line_points = P0 + np.outer(t, unit_dir)

    # 创建 3D 图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制无限长直线（虚线）
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
            color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Line through P0-P1')

    # 2. 强调原始线段 P0-P1（实线）
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]],
            color='blue', linewidth=3, label='Segment P0-P1')

    # 3. 绘制各个点
    ax.scatter(*P0, color='blue', s=80, label='P0', depthshade=False)
    ax.scatter(*P1, color='blue', s=80, label='P1', depthshade=False)
    ax.scatter(*aim_in_cam, color='red', s=100, label='Target (aim_in_cam)', depthshade=False)
    ax.scatter(*pj_pt, color='green', s=100, label='Projection (pj_pt)', depthshade=False)
    ax.scatter(*aim_in_cam1, color='purple', s=100, label='Symmetric Point (aim_in_cam1)', depthshade=False)

    # 4. 绘制垂线：aim_in_cam → pj_pt
    ax.plot([aim_in_cam[0], pj_pt[0]],
            [aim_in_cam[1], pj_pt[1]],
            [aim_in_cam[2], pj_pt[2]],
            color='green', linestyle='-', linewidth=2, label='Perpendicular')

    # 5. 绘制对称线：pj_pt → aim_in_cam1
    ax.plot([pj_pt[0], aim_in_cam1[0]],
            [pj_pt[1], aim_in_cam1[1]],
            [pj_pt[2], aim_in_cam1[2]],
            color='purple', linestyle='-', linewidth=2, label='Symmetric segment')

    # 6. 可选：从原点画辅助线（如果需要）
    # ax.plot([0, aim_in_cam[0]], [0, aim_in_cam[1]], [0, aim_in_cam[2]], color='black', alpha=0.3)

    # 设置标签和标题
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)

    # 设置等比例坐标轴
    all_points = np.array([P0, P1, aim_in_cam, pj_pt, aim_in_cam1, *line_points])
    X, Y, Z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(X), np.mean(Y), np.mean(Z)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def plot_coordinate_frames(rt1, rt2, frame_names=('Frame 1', 'Frame 2'), axis_length=1.0,
                           title="Two Coordinate Frames"):
    """
    绘制两个由 4x4 齐次变换矩阵定义的坐标系

    参数:
    - rt1, rt2: (4, 4) numpy 数组，齐次变换矩阵（旋转+平移）
    - frame_names: tuple of str, 坐标系名称
    - axis_length: float, 坐标轴显示长度
    - title: str, 图像标题
    """
    # 提取原点和旋转矩阵
    origin1 = rt1[:3, 3]
    R1 = rt1[:3, :3]

    origin2 = rt2[:3, 3]
    R2 = rt2[:3, :3]

    # 坐标轴方向（单位向量）
    axes = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])  # X, Y, Z

    # 变换到世界坐标系
    axes1 = (R1 @ axes.T).T * axis_length  # (3,3)
    axes2 = (R2 @ axes.T).T * axis_length  # (3,3)

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 Frame 1
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        # 轴线
        ax.quiver(*origin1, axes1[i, 0], axes1[i, 1], axes1[i, 2],
                  color=color, linewidth=2, label=f'{frame_names[0]}-{color.upper()}' if i == 0 else "")
        # 原点
        ax.scatter(*origin1, c='red', s=40, alpha=0.8)

    # 绘制 Frame 2
    for i, color in enumerate(colors):
        ax.quiver(*origin2, axes2[i, 0], axes2[i, 1], axes2[i, 2],
                  color=color, linewidth=2, linestyle='--', label=f'{frame_names[1]}-{color.upper()}' if i == 0 else "")
        ax.scatter(*origin2, c='blue', s=40, alpha=0.8)

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 添加图例（只加一次）
    if frame_names[0] != frame_names[1]:
        ax.legend()

    ax.set_title(title)

    # 设置等比例坐标轴
    all_pts = np.array([origin1, origin2, origin1 + axes1[0], origin1 + axes1[1], origin1 + axes1[2],
                        origin2 + axes2[0], origin2 + axes2[1], origin2 + axes2[2]])
    X, Y, Z = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2, (Z.max() + Z.min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()
