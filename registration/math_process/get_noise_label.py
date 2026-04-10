import torch
import torch.nn.functional as F




def euler_angles_to_rotation_matrix_zxy(euler_angles_deg):
    """
    ZXY 欧拉角转旋转矩阵 (PyTorch 批量版本)
    输入：(B, 3) 张量，单位：度 [z, x, y]
    输出：(B, 3, 3) 张量
    顺序：R = R_y @ R_x @ R_z
    """
    # 确保输入是 float 且在正确的设备上
    euler_angles_deg = euler_angles_deg.float()
    device = euler_angles_deg.device

    # 转换为弧度
    theta = torch.deg2rad(euler_angles_deg)
    theta_z = theta[:, 0]
    theta_x = theta[:, 1]
    theta_y = theta[:, 2]

    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)
    cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)

    batch_size = euler_angles_deg.shape[0]

    # 初始化旋转矩阵 (B, 3, 3)
    R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    # R_z
    R_z = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    R_z[:, 0, 0] = cos_z
    R_z[:, 0, 1] = -sin_z
    R_z[:, 1, 0] = sin_z
    R_z[:, 1, 1] = cos_z

    # R_x
    R_x = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    R_x[:, 1, 1] = cos_x
    R_x[:, 1, 2] = -sin_x
    R_x[:, 2, 1] = sin_x
    R_x[:, 2, 2] = cos_x

    # R_y
    R_y = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y

    # 矩阵乘法：R = R_y @ R_x @ R_z
    # torch.bmm 支持批量矩阵乘法
    R = torch.bmm(R_y, torch.bmm(R_x, R_z))

    return R


def rotation_matrix_to_euler_angles_zxy(R):
    """
    旋转矩阵转 ZXY 欧拉角 (PyTorch 批量版本)
    输入：(B, 3, 3) 张量
    输出：(B, 3) 张量，单位：度 [z, x, y]
    """
    device = R.device
    batch_size = R.shape[0]

    # 提取对应元素，保持维度 (B,)
    # 原 numpy 逻辑：sin_x = -R[1, 2]
    sin_x = -R[:, 1, 2]
    # 防止 arcsin 溢出
    sin_x = torch.clamp(sin_x, -1.0 + 1e-7, 1.0 - 1e-7)

    theta_x = torch.asin(sin_x)

    # 原 numpy 逻辑：theta_z = atan2(R[1, 0], R[1, 1])
    theta_z = torch.atan2(R[:, 1, 0], R[:, 1, 1])

    # 原 numpy 逻辑：theta_y = atan2(R[0, 2], R[2, 2])
    theta_y = torch.atan2(R[:, 0, 2], R[:, 2, 2])

    # 堆叠并转换为角度
    euler_rad = torch.stack([theta_z, theta_x, theta_y], dim=1)
    euler_deg = torch.rad2deg(euler_rad)

    return euler_deg


def get_transformer_noise(rotations, translations, R_ct2o):
    """
    计算相对于 O 的等效噪声（修正版）
    输入：
        rotations: (B, 3) tensor, degrees
        translations: (B, 3) tensor
        R_ct2o: (3, 3) tensor, constant rotation matrix
    输出：
        rot_noise_new: (B, 3) tensor, degrees
        trans_noise_new: (B, 3) tensor
    """
    device = rotations.device
    # 确保 R_ct2o 是 (3,3) 且类型正确
    R_ct2o_rot = R_ct2o[:3, :3].float().to(device)

    # ================= 1. 旋转变换（相似变换）=================
    # 1.1 CT 噪声欧拉角 -> 旋转矩阵 (B, 3, 3)
    R_noise_ct = euler_angles_to_rotation_matrix_zxy(rotations)

    # 1.2 相似变换：R_osz = R_ct2o @ R_ct @ R_ct2o.T
    # 由于 R_ct2o 是单矩阵，需要扩展维度或使用广播
    # R_noise_osz[b] = R_ct2o @ R_noise_ct[b] @ R_ct2o.T
    R_ct2o_T = R_ct2o_rot.T

    # 使用 bmm 进行批量计算：(B, 3, 3) @ (3, 3) -> 需要调整
    # 方法：R_noise_ct @ R_ct2o_T 得到 (B, 3, 3)
    # 然后 R_ct2o @ Result -> 需要 R_ct2o 扩展为 (B, 3, 3)
    R_ct2o_batch = R_ct2o_rot.unsqueeze(0).expand(rotations.shape[0], -1, -1)

    temp = torch.bmm(R_noise_ct, R_ct2o_T.unsqueeze(0).expand(rotations.shape[0], -1, -1))
    R_noise_osz = torch.bmm(R_ct2o_batch, temp)

    # 1.3 旋转矩阵 -> OSZ 噪声欧拉角
    rot_noise_new = rotation_matrix_to_euler_angles_zxy(R_noise_osz)

    # ================= 2. 平移变换（向量变换）=================
    # 原逻辑：trans_noise_new = R_ct2o_rot @ translations
    # 由于 translations 是 (B, 3) 行向量，数学上等价于 t_new = t_old @ R.T
    trans_noise_new = torch.matmul(translations, R_ct2o_T)

    return rot_noise_new, trans_noise_new




def euler_to_rotvec(euler_deg):
    """
    ZXY 欧拉角 → 旋转向量（单位：度）
    支持批量输入 [B, 3] 或单样本 [3]
    """
    # 确保输入是 torch 张量
    if not isinstance(euler_deg, torch.Tensor):
        euler_deg = torch.tensor(euler_deg, dtype=torch.float32)

    # 保存原始形状，支持 [3] 或 [B, 3]
    original_shape = euler_deg.shape
    if euler_deg.dim() == 1:
        euler_deg = euler_deg.unsqueeze(0)  # [3] -> [1, 3]

    batch_size = euler_deg.shape[0]
    device = euler_deg.device
    dtype = euler_deg.dtype

    # ================= 1. 欧拉角 → 旋转矩阵 (ZXY 序) =================
    theta_z = torch.deg2rad(euler_deg[:, 0])
    theta_x = torch.deg2rad(euler_deg[:, 1])
    theta_y = torch.deg2rad(euler_deg[:, 2])

    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)
    cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)

    # 构建旋转矩阵 (批量)
    R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)

    # R = R_y @ R_x @ R_z (ZXY 序)
    R[:, 0, 0] = cos_y * cos_z - sin_y * sin_x * sin_z
    R[:, 0, 1] = -cos_y * sin_z - sin_y * sin_x * cos_z
    R[:, 0, 2] = sin_y * cos_x

    R[:, 1, 0] = cos_x * sin_z
    R[:, 1, 1] = cos_x * cos_z
    R[:, 1, 2] = -sin_x

    R[:, 2, 0] = -sin_y * cos_z - cos_y * sin_x * sin_z
    R[:, 2, 1] = sin_y * sin_z - cos_y * sin_x * cos_z
    R[:, 2, 2] = cos_y * cos_x

    # ================= 2. 旋转矩阵 → 旋转向量 =================
    # 计算旋转角：angle = arccos((trace(R) - 1) / 2)
    trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # [B]
    angle = torch.acos(torch.clamp((trace_R - 1) / 2, -1 + 1e-7, 1 - 1e-7))  # [B]

    # 处理小角度情况
    small_angle_mask = angle < 1e-8  # [B]

    # 计算旋转轴
    axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    axis[:, 0] = R[:, 2, 1] - R[:, 1, 2]
    axis[:, 1] = R[:, 0, 2] - R[:, 2, 0]
    axis[:, 2] = R[:, 1, 0] - R[:, 0, 1]

    # 归一化旋转轴 (避免除零)
    sin_angle = torch.sin(angle)
    axis = axis / (2 * sin_angle + 1e-12).unsqueeze(-1)  # [B, 3]

    # 小角度时直接返回 0 向量
    rot_vec = torch.rad2deg(angle).unsqueeze(-1) * axis  # [B, 3]
    rot_vec[small_angle_mask] = 0.0

    # 恢复原始形状
    if len(original_shape) == 1:
        rot_vec = rot_vec.squeeze(0)

    return rot_vec


def rotvec_to_euler(rotvec_deg):
    """
    旋转向量 → ZXY 欧拉角（单位：度）
    支持批量输入 [B, 3] 或单样本 [3]
    """
    # 确保输入是 torch 张量
    if not isinstance(rotvec_deg, torch.Tensor):
        rotvec_deg = torch.tensor(rotvec_deg, dtype=torch.float32)

    # 保存原始形状
    original_shape = rotvec_deg.shape
    if rotvec_deg.dim() == 1:
        rotvec_deg = rotvec_deg.unsqueeze(0)  # [3] -> [1, 3]

    batch_size = rotvec_deg.shape[0]
    device = rotvec_deg.device
    dtype = rotvec_deg.dtype

    # ================= 1. 旋转向量 → 旋转角 + 旋转轴 =================
    angle = torch.rad2deg(torch.norm(rotvec_deg, dim=1))  # [B], 单位：度
    angle_rad = torch.deg2rad(angle)  # [B], 单位：弧度

    # 处理小角度情况
    small_angle_mask = angle_rad < 1e-8  # [B]

    # 计算旋转轴 (归一化)
    axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    norm_vec = torch.norm(rotvec_deg, dim=1, keepdim=True) + 1e-12  # [B, 1]
    axis[~small_angle_mask] = rotvec_deg[~small_angle_mask] / norm_vec[~small_angle_mask]

    # ================= 2. 旋转轴 + 角 → 旋转矩阵 (Rodrigues 公式) =================
    R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)

    # 预计算三角函数
    sin_angle = torch.sin(angle_rad)  # [B]
    cos_angle = torch.cos(angle_rad)  # [B]
    one_minus_cos = 1 - cos_angle  # [B]

    # 构建反对称矩阵 K 的元素
    K01 = -axis[:, 2]  # [B]
    K02 = axis[:, 1]
    K12 = -axis[:, 0]

    # Rodrigues 公式：R = I + sin(θ)*K + (1-cos(θ))*K@K
    # 直接展开计算 (批量)
    R[:, 0, 0] = 1 + one_minus_cos * axis[:, 0] * axis[:, 0]
    R[:, 0, 1] = one_minus_cos * axis[:, 0] * axis[:, 1] - sin_angle * axis[:, 2]
    R[:, 0, 2] = one_minus_cos * axis[:, 0] * axis[:, 2] + sin_angle * axis[:, 1]

    R[:, 1, 0] = one_minus_cos * axis[:, 1] * axis[:, 0] + sin_angle * axis[:, 2]
    R[:, 1, 1] = 1 + one_minus_cos * axis[:, 1] * axis[:, 1]
    R[:, 1, 2] = one_minus_cos * axis[:, 1] * axis[:, 2] - sin_angle * axis[:, 0]

    R[:, 2, 0] = one_minus_cos * axis[:, 2] * axis[:, 0] - sin_angle * axis[:, 1]
    R[:, 2, 1] = one_minus_cos * axis[:, 2] * axis[:, 1] + sin_angle * axis[:, 0]
    R[:, 2, 2] = 1 + one_minus_cos * axis[:, 2] * axis[:, 2]

    # 小角度时设为单位矩阵
    R[small_angle_mask] = torch.eye(3, device=device, dtype=dtype)

    # ================= 3. 旋转矩阵 → ZXY 欧拉角 =================
    euler = torch.zeros(batch_size, 3, device=device, dtype=dtype)

    # theta_x = arcsin(-R[1, 2])
    sin_x = -R[:, 1, 2]
    sin_x = torch.clamp(sin_x, -1 + 1e-7, 1 - 1e-7)
    theta_x = torch.asin(sin_x)  # [B]

    # theta_z = arctan2(R[1, 0], R[1, 1])
    theta_z = torch.atan2(R[:, 1, 0], R[:, 1, 1])  # [B]

    # theta_y = arctan2(R[0, 2], R[2, 2])
    theta_y = torch.atan2(R[:, 0, 2], R[:, 2, 2])  # [B]

    # 组合并转换为角度
    euler[:, 0] = torch.rad2deg(theta_z)
    euler[:, 1] = torch.rad2deg(theta_x)
    euler[:, 2] = torch.rad2deg(theta_y)

    # 小角度时返回 0
    euler[small_angle_mask] = 0.0

    # 恢复原始形状
    if len(original_shape) == 1:
        euler = euler.squeeze(0)

    return euler


def get_transformer_noise_vector(rotations, translations, R_ct2o):
    """
    基于旋转向量的噪声变换 (PyTorch 版本)
    输入:
        rotations: [B, 3] 或 [3] 欧拉角 (度)
        translations: [B, 3] 或 [3] 平移
        R_ct2o: [3, 3] 或 [B, 3, 3] 变换矩阵
    输出:
        rot_noise_new: [B, 3] 或 [3] 欧拉角 (度)
        trans_noise_new: [B, 3] 或 [3] 平移
    """
    # 确保输入是 torch 张量
    if not isinstance(rotations, torch.Tensor):
        rotations = torch.tensor(rotations, dtype=torch.float32)
    if not isinstance(translations, torch.Tensor):
        translations = torch.tensor(translations, dtype=torch.float32)
    if not isinstance(R_ct2o, torch.Tensor):
        R_ct2o = torch.tensor(R_ct2o, dtype=torch.float32)

    # 处理维度
    single_sample = rotations.dim() == 1
    if single_sample:
        rotations = rotations.unsqueeze(0)
        translations = translations.unsqueeze(0)

    batch_size = rotations.shape[0]
    device = rotations.device

    # 扩展 R_ct2o 到批量维度
    if R_ct2o.dim() == 2:
        R_ct2o = R_ct2o.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 3, 3]

    R_ct2o_rot = R_ct2o[:, :3, :3]  # [B, 3, 3]

    # ================= 1. 旋转变换 =================
    rot_vec_ct = euler_to_rotvec(rotations)  # [B, 3]
    rot_vec_osz = torch.bmm(R_ct2o_rot, rot_vec_ct.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    rot_noise_new = rotvec_to_euler(rot_vec_osz)  # [B, 3]

    # ================= 2. 平移变换 =================
    trans_noise_new = torch.bmm(R_ct2o_rot, translations.unsqueeze(-1)).squeeze(-1)  # [B, 3]

    # 恢复单样本维度
    if single_sample:
        rot_noise_new = rot_noise_new.squeeze(0)
        trans_noise_new = trans_noise_new.squeeze(0)

    return rot_noise_new, trans_noise_new