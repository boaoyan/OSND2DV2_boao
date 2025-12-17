import numpy as np

from robot_arm.utils.cali_utils import transform_points


# 已知A图中一个像素点的位置，求解其在B图的直线
def get_line(u, v, rt_ct2o_s1, rt_ct2o_s2, a_arm, a_inv, L=800):
    """
    获取像素点 (u, v) 在另一视图上的投影直线
    参数说明：
        u, v: 当前视图的像素坐标
        rt_ct2o_s1: 第一个坐标系（体素->光源）变换矩阵
        rt_ct2o_s2: 第二个坐标系（体素->光源）变换矩阵
        a_arm: 相机内参矩阵
        a_inv: 光源归一化矩阵
        L: 光源到像平面的距离
    返回值：
        m, c：投影在另一视图上的直线 y = mx + c 的参数
    """
    # 当前视图下像素点对应的3D光线方向（光源坐标系）
    oz_xyz1 = a_inv @ np.array([L * u, L * v, L])
    oz_xyz2 = np.zeros(3)  # 另一个点为光源原点

    # 求逆矩阵：光源 -> 体素坐标
    rt_o2ct_s1 = np.linalg.inv(rt_ct2o_s1)

    # 将两点从光源坐标 -> 体素坐标（使用第一个视图的反变换）
    ct_xyz1 = rt_o2ct_s1 @ np.append(oz_xyz1, 1)
    ct_xyz2 = rt_o2ct_s1 @ np.append(oz_xyz2, 1)

    # 将体素坐标 -> 投影到第二个视图的光源坐标中
    oc_xyz1 = rt_ct2o_s2 @ ct_xyz1
    oc_xyz2 = rt_ct2o_s2 @ ct_xyz2

    # 使用第二个视图的相机参数转换到像素坐标
    pc1 = a_arm @ oc_xyz1[:3]
    pc2 = a_arm @ oc_xyz2[:3]

    puc1, pvc1 = pc1[0] / pc1[2], pc1[1] / pc1[2]
    puc2, pvc2 = pc2[0] / pc2[2], pc2[1] / pc2[2]

    # 计算直线方程
    m, c = line_equation(puc1, pvc1, puc2, pvc2)
    return m, c


def line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)  # 斜率
    c = y1 - m * x1  # y轴截距
    return m, c


def get_pixel_from_ct(ct_point, rt_ct2o_s1, rt_ct2o_s2, a_arm):
    """
    将CT体素空间中的一个点反向投影到两个视图的像素坐标(u1, v1), (u2, v2)

    参数:
        ct_point: CT体素空间中的3D点 (x, y, z)
        rt_ct2o_s1: 第一视角的体素->光源坐标变换矩阵
        rt_ct2o_s2: 第二视角的体素->光源坐标变换矩阵
        a_arm: 相机内参矩阵（像素映射）

    返回:
        (u1, v1): 在第一视角下的像素坐标
        (u2, v2): 在第二视角下的像素坐标
    """
    ct_point_homogeneous = np.append(ct_point, 1)  # 转为齐次坐标

    # 将CT点变换到两个视图的光源坐标系中
    o_xyz_s1 = rt_ct2o_s1 @ ct_point_homogeneous
    o_xyz_s2 = rt_ct2o_s2 @ ct_point_homogeneous

    # 去掉齐次维度
    o_xyz_s1 = o_xyz_s1[:3]
    o_xyz_s2 = o_xyz_s2[:3]

    # 使用光源归一化矩阵得到像素方向
    dir_s1 = a_arm @ o_xyz_s1
    dir_s2 = a_arm @ o_xyz_s2

    # 归一化方向向量（除以L）
    u1, v1 = dir_s1[0] / dir_s1[2], dir_s1[1] / dir_s1[2]
    u2, v2 = dir_s2[0] / dir_s2[2], dir_s2[1] / dir_s2[2]

    return np.array([u1, v1]), np.array([u2, v2])

def get_point_in_ct(u, v, rt_ct2o, a_inv, L=800):

    rt_o2ct = np.linalg.inv(rt_ct2o)
    # 假设光源零点
    o_source = np.zeros(3)
    # 从光源坐标到CT体素坐标
    oct_source = rt_o2ct @ np.append(o_source, 1).T
    # 从像素坐标到各自的光源坐标
    pot = a_inv @ np.array([L * u, L * v, L]).T
    # 从各自的光源坐标统一到CT体素坐标
    pct = rt_o2ct @ np.append(pot, 1).T

    return oct_source[:3], pct[:3]


def get_coord_in_ct(uv1, uv2, rt_ct2o_sz, rt_ct2o_sc, a_inv, L=800):
    u1, v1 = uv1
    u2, v2 = uv2
    rt_o2ct_sz = np.linalg.inv(rt_ct2o_sz)
    rt_o2ct_sc = np.linalg.inv(rt_ct2o_sc)
    # 假设所有的光源零点
    o_source_sz = np.zeros(3)
    o_source_sc = np.zeros(3)
    # 从光源坐标到CT体素坐标
    oct_source_sz = rt_o2ct_sz @ np.append(o_source_sz, 1).T
    oct_source_sc = rt_o2ct_sc @ np.append(o_source_sc, 1).T
    # 从像素坐标到各自的光源坐标
    pot_sz = a_inv @ np.array([L * u1, L * v1, L]).T
    pot_sc = a_inv @ np.array([L * u2, L * v2, L]).T
    # 从各自的光源坐标统一到CT体素坐标
    pct_sz = rt_o2ct_sz @ np.append(pot_sz, 1).T
    pct_sc = rt_o2ct_sc @ np.append(pot_sc, 1).T
    # 已知空间中的两条线，求一个点到两条线的距离最短
    d1 = pct_sz - oct_source_sz
    d2 = pct_sc - oct_source_sc
    res = intersection_of_multi_lines(np.array([pct_sz[:3], pct_sc[:3]]), np.array([d1[:3], d2[:3]]))
    return res[:3]


def get_origin_sz_in_body(rt_o2tool_sz, rt_tool2cam_sz, rt_body2cam_sz):
    rt_o2cam_sz = rt_tool2cam_sz @ rt_o2tool_sz
    rt_cam2body_sz = np.linalg.inv(rt_body2cam_sz)
    rt_o2body_sz = rt_cam2body_sz @ rt_o2cam_sz
    return rt_o2body_sz


def get_origin_sc_in_body(rt_o2tool_sc, rt_tool2cam_sc, rt_body2cam_sc):
    rt_o2cam_sc = rt_tool2cam_sc @ rt_o2tool_sc
    rt_cam2body_sc = np.linalg.inv(rt_body2cam_sc)
    rt_o2body_sc = rt_cam2body_sc @ rt_o2cam_sc
    return rt_o2body_sc

def get_bodysz2bodysc(rt_body2cam_sz, rt_body2cam_sc):
    rt_cam2body_sc = np.linalg.inv(rt_body2cam_sc)
    rt_bodysz2bodysc = rt_cam2body_sc @ rt_body2cam_sz
    return rt_bodysz2bodysc

def get_coord_in_ct_updata(uv1, uv2, rt_o2body_sz, rt_o2body_sc, rt_bodysz2bodysc, a_inv, L=800):
    u1, v1 = uv1
    u2, v2 = uv2
    # 假设所有的光源零点
    o_source_sz = np.zeros(3)
    o_source_sc = np.zeros(3)
    # 从光源坐标到人体模版坐标
    oct_source_sz = rt_o2body_sz @ np.append(o_source_sz, 1).T
    oct_source_sz_update = rt_bodysz2bodysc @ np.append(oct_source_sz)
    oct_source_sc = rt_o2body_sc @ np.append(o_source_sc, 1).T
    # 从像素坐标到各自的光源坐标
    pot_sz = a_inv @ np.array([L * u1, L * v1, L]).T
    pot_sc = a_inv @ np.array([L * u2, L * v2, L]).T
    # 从各自的光源坐标统一到CT体素坐标
    pct_sz = rt_o2body_sz @ np.append(pot_sz, 1).T
    pct_sz_updata = rt_bodysz2bodysc @ np.append(pct_sz)
    pct_sc = rt_o2body_sc @ np.append(pot_sc, 1).T
    # 已知空间中的两条线，求一个点到两条线的距离最短
    d1 = pct_sz_updata - oct_source_sz_update
    d2 = pct_sc - oct_source_sc
    res = intersection_of_multi_lines(np.array([pct_sz[:3], pct_sc[:3]]), np.array([d1[:3], d2[:3]]))
    return res[:3]

def get_coord_in_ct_updata_new(uv1, uv2, rt_ct2o_sz, rt_ct2o_sc, a_inv, L=800):
    u1, v1 = uv1
    u2, v2 = uv2
    rt_tool2body_sz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_tool2body_sc = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_offset = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_osz2osc = rt_ct2o_sc @ np.linalg.inv(rt_ct2o_sz)
    rt_body2o_sz = rt_ct2o_sz
    rt_body2o_sc = rt_ct2o_sc @ rt_offset
    rt_o2tool_sz = np.linalg.inv(rt_body2o_sz @ rt_tool2body_sz)
    rt_o2tool_sc = np.linalg.inv(rt_body2o_sc @ rt_tool2body_sc)

    rt_o2body_sz = rt_tool2body_sz @ rt_o2tool_sz
    rt_o2body_sc = rt_tool2body_sc @ rt_o2tool_sc
    rt_bodysz2bodysc = rt_o2body_sc @ rt_osz2osc @ np.linalg.inv(rt_o2body_sz)
    # 假设所有的光源零点
    o_source_sz = np.zeros(3)
    o_source_sc = np.zeros(3)
    # 从光源坐标到人体模版坐标
    oct_source_sz = rt_o2body_sz @ np.append(o_source_sz, 1).T
    oct_source_sz_update = rt_bodysz2bodysc @ np.append(oct_source_sz)
    oct_source_sc = rt_o2body_sc @ np.append(o_source_sc, 1).T
    # 从像素坐标到各自的光源坐标
    pot_sz = a_inv @ np.array([L * u1, L * v1, L]).T
    pot_sc = a_inv @ np.array([L * u2, L * v2, L]).T
    # 从各自的光源坐标统一到CT体素坐标
    pct_sz = rt_o2body_sz @ np.append(pot_sz, 1).T
    pct_sz_updata = rt_bodysz2bodysc @ np.append(pct_sz)
    pct_sc = rt_o2body_sc @ np.append(pot_sc, 1).T
    # 已知空间中的两条线，求一个点到两条线的距离最短
    d1 = pct_sz_updata - oct_source_sz_update
    d2 = pct_sc - oct_source_sc
    res = intersection_of_multi_lines(np.array([pct_sz[:3], pct_sc[:3]]), np.array([d1[:3], d2[:3]]))
    return res[:3]

def intersection_of_multi_lines(start_points, directions):
    '''
    # https://zhuanlan.zhihu.com/p/146190385
    @param start_points: line start points; numpy array, nxdim
    @param directions: list directions; numpy array, nxdim
    @return: the nearest points to n lines
    '''

    n, dim = start_points.shape

    G_left = np.tile(np.eye(dim), (n, 1))
    G_right = np.zeros((dim * n, n))

    for i in range(n):
        G_right[i * dim:(i + 1) * dim, i] = -directions[i, :]

    G = np.concatenate([G_left, G_right], axis=1)
    d = start_points.reshape((-1, 1))

    m = np.linalg.inv(np.dot(G.T, G)).dot(G.T).dot(d)

    return m[0:dim].flatten()


def get_pj_pt(aim_pt, P0, P1):
    # 计算方向向量
    direction = P1 - P0
    length = np.linalg.norm(direction)
    unit_dir = direction / length
    # 计算最近点（投影）
    point_vec = aim_pt - P0
    t_proj = np.dot(point_vec, unit_dir)
    closest_point = P0 + t_proj * unit_dir
    return closest_point
