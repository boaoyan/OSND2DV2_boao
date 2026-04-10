import numpy as np


def get_base_noise(rot_noise_new, trans_noise_new, R_ct2o):
    """
    逆变换：OSZ 系等效噪声 → CT 系原始噪声 (支持批量)
    修正：补全转置逻辑，移除强制对齐干扰
    """
    rotations = np.asarray(rot_noise_new, dtype=np.float64)
    translations = np.asarray(trans_noise_new, dtype=np.float64)
    R_ct2o = np.asarray(R_ct2o, dtype=np.float64)

    # 兼容单样本 (3,) 自动升维为 (1, 3)
    squeeze_out = False
    if rotations.ndim == 1:
        rotations = rotations[np.newaxis, :]
        translations = translations[np.newaxis, :]
        squeeze_out = True

    R_ct2o_rot = R_ct2o[:3, :3]  # (3, 3)

    # ================= 1. 旋转逆变换（严格相似变换的逆）=================
    R_noise_osz = euler_angles_to_rotation_matrix_zxy(rotations)  # [N, 3, 3]

    # ✅ 关键修正：首项必须为 R_ct2o_rot.T，对应数学公式 R^T @ R_noise @ R
    R_noise_ct = np.einsum('ij,njk,kl->nil', R_ct2o_rot.T, R_noise_osz, R_ct2o_rot)

    # 旋转矩阵 → CT 系欧拉角 [N, 3]
    rotations_ct = rotation_matrix_to_euler_angles_zxy_numpy(R_noise_ct)

    # ================= 2. 平移逆变换（向量变换的逆）=================
    translations_ct = np.einsum('ij,nj->ni', R_ct2o_rot.T, translations)

    # 恢复单样本形状
    if squeeze_out:
        rotations_ct = rotations_ct.squeeze(0)
        translations_ct = translations_ct.squeeze(0)

    return rotations_ct, translations_ct


def euler_angles_to_rotation_matrix_zxy(euler_angles_deg):
    """ZXY 欧拉角转旋转矩阵 (支持批量 [N, 3] -> [N, 3, 3])"""
    theta = np.deg2rad(euler_angles_deg)  # [N, 3]
    cz, sz = np.cos(theta[:, 0]), np.sin(theta[:, 0])
    cx, sx = np.cos(theta[:, 1]), np.sin(theta[:, 1])
    cy, sy = np.cos(theta[:, 2]), np.sin(theta[:, 2])

    R = np.zeros((euler_angles_deg.shape[0], 3, 3), dtype=np.float64)

    # 解析展开 R = R_y @ R_x @ R_z 的 9 个元素
    R[:, 0, 0] = cz * cy + sz * sx * sy
    R[:, 0, 1] = -cy * sz + sy * sx * cz
    R[:, 0, 2] = sy * cx
    R[:, 1, 0] = cx * sz
    R[:, 1, 1] = cx * cz
    R[:, 1, 2] = -sx
    R[:, 2, 0] = -sy * cz + cy * sx * sz
    R[:, 2, 1] = sy * sz + cy * sx * cz
    R[:, 2, 2] = cy * cx
    return R


def rotation_matrix_to_euler_angles_zxy_numpy(R):
    """旋转矩阵转 ZXY 欧拉角 (支持批量 [N, 3, 3] -> [N, 3])"""
    # 提取 sin(x)，并限制范围防止浮点误差导致 arcsin 报错
    sin_x = -np.clip(R[:, 1, 2], -1.0, 1.0)
    theta_x = np.arcsin(sin_x)

    # 提取 Z 和 Y 角 (使用 atan2 保证象限正确)
    theta_z = np.arctan2(R[:, 1, 0], R[:, 1, 1])
    theta_y = np.arctan2(R[:, 0, 2], R[:, 2, 2])

    return np.rad2deg(np.column_stack((theta_z, theta_x, theta_y)))



def get_source_transform(pa_rota, pa_trans, rlat_rota, rlat_trans, R_ctsz2osz, R_ctsc2osc):
    """
    计算正位与侧位光源在加入CT噪声后的相对变换矩阵 R_sz_osc2osz

    公式: R_sz_osc2osz = R_ctsz2osz @ inv(delta_T_PA) @ delta_T_RLAT @ inv(R_ctsc2osc)

    Args:
        pa_rota: [N, 3] 或 [3] 正位CT旋转噪声 (度, ZXY)
        pa_trans: [N, 3] 或 [3] 正位CT平移噪声 (mm)
        rlat_rota: [N, 3] 或 [3] 侧位CT旋转噪声 (度, ZXY)
        rlat_trans: [N, 3] 或 [3] 侧位CT平移噪声 (mm)
        R_ctsz2osz: [4, 4] 标准CT→正位光源变换矩阵
        R_ctsc2osc: [4, 4] 标准CT→侧位光源变换矩阵

    Returns:
        R_sz_osc2osz: [N, 4, 4] 或 [4, 4] 加噪后侧位光源→正位光源变换矩阵
    """
    # 1. 统一数据类型
    pa_rota = np.asarray(pa_rota, dtype=np.float64)
    pa_trans = np.asarray(pa_trans, dtype=np.float64)
    rlat_rota = np.asarray(rlat_rota, dtype=np.float64)
    rlat_trans = np.asarray(rlat_trans, dtype=np.float64)
    R_ctsz2osz = np.asarray(R_ctsz2osz, dtype=np.float64)
    R_ctsc2osc = np.asarray(R_ctsc2osc, dtype=np.float64)

    # 兼容单样本 (3,) -> (1, 3)
    squeeze_out = False
    if pa_rota.ndim == 1:
        pa_rota = pa_rota[np.newaxis, :]
        pa_trans = pa_trans[np.newaxis, :]
        rlat_rota = rlat_rota[np.newaxis, :]
        rlat_trans = rlat_trans[np.newaxis, :]
        squeeze_out = True

    batch_size = pa_rota.shape[0]

    # 2. 构建正位CT噪声齐次变换矩阵 Delta_T_PA (N, 4, 4)
    R_noise_pa = euler_angles_to_rotation_matrix_zxy(pa_rota)
    delta_T_PA = np.zeros((batch_size, 4, 4), dtype=np.float64)
    delta_T_PA[:, :3, :3] = R_noise_pa
    delta_T_PA[:, :3, 3] = pa_trans
    delta_T_PA[:, 3, 3] = 1.0

    # 3. 构建侧位CT噪声齐次变换矩阵 Delta_T_RLAT (N, 4, 4)
    R_noise_rlat = euler_angles_to_rotation_matrix_zxy(rlat_rota)
    delta_T_RLAT = np.zeros((batch_size, 4, 4), dtype=np.float64)
    delta_T_RLAT[:, :3, :3] = R_noise_rlat
    delta_T_RLAT[:, :3, 3] = rlat_trans
    delta_T_RLAT[:, 3, 3] = 1.0

    # 4. 预计算逆矩阵
    inv_delta_T_PA = np.linalg.inv(delta_T_PA)  # (N, 4, 4)
    inv_R_ctsc2osc = np.linalg.inv(R_ctsc2osc)  # (4, 4)

    # 5. 核心公式链乘
    # NumPy @ 运算符原生支持 (4,4) @ (N,4,4) @ (N,4,4) @ (4,4) 的自动广播
    # 计算流: (4,4)@(N,4,4)->(N,4,4) @(N,4,4)->(N,4,4) @(4,4)->(N,4,4)
    R_sz_osc2osz = R_ctsz2osz @ inv_delta_T_PA @ delta_T_RLAT @ inv_R_ctsc2osc

    # 6. 恢复单样本形状
    if squeeze_out:
        R_sz_osc2osz = R_sz_osc2osz.squeeze(0)

    return R_sz_osc2osz




def get_osz_noise(rlat_rota, rlat_trans, R_sz_osc2osz, R_ctsz2osz, R_ctsc2osc):
    """
    将侧位CT视角噪声转换到正位光源视角下

    公式: delta_T_sz = delta_T_sc @ inv_R_ctsc2osc @ inv_R_sz_osc2osz @ R_ctsz2osz
    """
    # 1. 统一数据类型
    rota_noise = np.asarray(rlat_rota, dtype=np.float64)
    trans_noise = np.asarray(rlat_trans, dtype=np.float64)
    R_sz_osc2osz = np.asarray(R_sz_osc2osz, dtype=np.float64)
    R_ctsz2osz = np.asarray(R_ctsz2osz, dtype=np.float64)
    R_ctsc2osc = np.asarray(R_ctsc2osc, dtype=np.float64)

    # 2. 兼容单样本 (3,) -> (1, 3)
    squeeze_out = False
    if rota_noise.ndim == 1:
        rota_noise = rota_noise[np.newaxis, :]
        trans_noise = trans_noise[np.newaxis, :]
        squeeze_out = True

    batch_size = rota_noise.shape[0]

    # 3. 构建侧位CT噪声齐次变换矩阵 Delta_T_sc (N, 4, 4)
    R_noise_sc = euler_angles_to_rotation_matrix_zxy(rota_noise)
    delta_T_sc = np.zeros((batch_size, 4, 4), dtype=np.float64)
    delta_T_sc[:, :3, :3] = R_noise_sc
    delta_T_sc[:, :3, 3] = trans_noise
    delta_T_sc[:, 3, 3] = 1.0

    # 4. 预计算逆矩阵 (4, 4)
    inv_R_ctsc2osc = np.linalg.inv(R_ctsc2osc)
    inv_R_sz_osc2osz = np.linalg.inv(R_sz_osc2osz)
    # 5. 核心公式链式计算
    # @ 运算符原生支持 (N,4,4) @ (4,4) 或 (N,4,4) @ (N,4,4) 的自动广播
    # 彻底避免 einsum 下标不匹配问题
    delta_T_sz = delta_T_sc @ inv_R_ctsc2osc @ inv_R_sz_osc2osz @ R_ctsz2osz

    # 6. 【逆运算提取】还原为正位视角噪声
    pa_rota = rotation_matrix_to_euler_angles_zxy_numpy(delta_T_sz[:, :3, :3])
    pa_trans = delta_T_sz[:, :3, 3]

    # 7. 恢复单样本形状
    if squeeze_out:
        pa_rota = pa_rota.squeeze(0)
        pa_trans = pa_trans.squeeze(0)

    return pa_rota, pa_trans


