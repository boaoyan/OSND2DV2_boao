import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from tqdm import tqdm


def euler_zxy_to_rot_batch(angles_deg: torch.Tensor) -> torch.Tensor:
    """批量将ZXY欧拉角(度)转为旋转矩阵 [B, 3, 3]"""
    rad = angles_deg * (torch.pi / 180.0)
    z, x, y = rad[:, 0], rad[:, 1], rad[:, 2]
    c, s = torch.cos, torch.sin

    Rz = torch.stack([
        torch.stack([c(z), -s(z), torch.zeros_like(z)], dim=1),
        torch.stack([s(z), c(z), torch.zeros_like(z)], dim=1),
        torch.stack([torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)], dim=1)
    ], dim=1)

    Rx = torch.stack([
        torch.stack([torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=1),
        torch.stack([torch.zeros_like(x), c(x), -s(x)], dim=1),
        torch.stack([torch.zeros_like(x), s(x), c(x)], dim=1)
    ], dim=1)

    Ry = torch.stack([
        torch.stack([c(y), torch.zeros_like(y), s(y)], dim=1),
        torch.stack([torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y)], dim=1),
        torch.stack([-s(y), torch.zeros_like(y), c(y)], dim=1)
    ], dim=1)

    return Ry @ Rx @ Rz  # [B, 3, 3]


def compute_full_voxel_pose_loss(
        nifti_path: str,
        csv_path: str,
        output_csv_path: str = None,
        intensity_threshold: float = -1000.0,  # CT空气阈值，过滤无效背景
        point_batch_size: int = 200_000,  # 单次加载的体素点数（控制显存）
        sample_batch_size: int = 64,  # 单次计算的样本/步数批次
        device: str = "cuda"
) -> pd.DataFrame:
    """
    基于全量体素计算位姿空间误差 (MSE, 单位: mm²)

    :param nifti_path: .nii.gz 体素文件路径
    :param csv_path: 包含 pre_*/tru_* 列的CSV路径
    :param intensity_threshold: 强度阈值，低于此值的体素将被忽略（默认过滤CT空气）
    :param point_batch_size: 显存安全切分大小
    :param sample_batch_size: 样本并行计算大小
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"📖 1. 加载NIfTI体素: {nifti_path}")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = torch.tensor(img.affine, dtype=torch.float32, device=device)

    # 提取有效体素索引 (numpy argwhere 返回 [z, y, x])
    valid_mask = data > intensity_threshold
    indices = np.argwhere(valid_mask).astype(np.float32)
    print(f"   📊 有效体素数量: {len(indices):,}")

    # ⚠️ NIfTI仿射矩阵期望输入顺序为 [x, y, z, 1]，需交换列顺序
    indices_xyz = indices[:, [2, 1, 0]]
    ones = np.ones((len(indices_xyz), 1), dtype=np.float32)
    pts_hom = np.concatenate([indices_xyz, ones], axis=1)

    # 转换到物理坐标 (mm)
    pts_phys = torch.tensor(pts_hom, device=device) @ affine.T
    pts_phys = pts_phys[:, :3]  # [N, 3]
    N = pts_phys.shape[0]

    print(f"📖 2. 读取位姿CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    gt_cols = ['tru_rota_z', 'tru_rota_x', 'tru_rota_y', 'tru_trans_x', 'tru_trans_y', 'tru_trans_z']
    pred_cols = ['pre_rota_z', 'pre_rota_x', 'pre_rota_y', 'pre_trans_x', 'pre_trans_y', 'pre_trans_z']

    gt_poses = torch.tensor(df[gt_cols].values, device=device, dtype=torch.float32)
    pred_poses = torch.tensor(df[pred_cols].values, device=device, dtype=torch.float32)
    M = len(df)

    # 预计算所有样本的旋转矩阵与平移向量
    R_gt = euler_zxy_to_rot_batch(gt_poses[:, :3])
    R_pred = euler_zxy_to_rot_batch(pred_poses[:, :3])
    t_gt = gt_poses[:, 3:]
    t_pred = pred_poses[:, 3:]

    print(f"⚡ 3. 开始计算全量体素Loss | 样本数: {M:,} | 体素数: {N:,}")
    losses = torch.zeros(M, device=device)

    # 🔄 双重循环：外层样本批次，内层体素批次（显存友好 + GPU加速）
    for s_start in tqdm(range(0, M, sample_batch_size), desc="样本批次"):
        s_end = min(s_start + sample_batch_size, M)
        R_diff = R_gt[s_start:s_end] - R_pred[s_start:s_end]  # [B_s, 3, 3]
        t_diff = t_gt[s_start:s_end] - t_pred[s_start:s_end]  # [B_s, 3]
        B_s = s_end - s_start

        batch_loss = torch.zeros(B_s, device=device)

        for p_start in range(0, N, point_batch_size):
            p_end = min(p_start + point_batch_size, N)
            pts_b = pts_phys[p_start:p_end]  # [B_p, 3]

            # 核心公式: Δp = (R_gt - R_pred) @ p + (t_gt - t_pred)
            # pts_b.T: [3, B_p] -> bmm后得 [B_s, 3, B_p]
            disp = torch.bmm(R_diff, pts_b.T.permute(1, 0)) + t_diff.unsqueeze(-1)

            # 累加平方误差 (对3个维度与B_p个点求和)
            batch_loss += (disp ** 2).sum(dim=[1, 2])

        # 归一化为平均MSE (单位: mm²)
        losses[s_start:s_end] = batch_loss / (N * 3.0)

    # 💾 保存结果
    df['voxel_mse_loss'] = losses.cpu().numpy()
    if output_csv_path is None:
        base, ext = os.path.splitext(csv_path)
        output_csv_path = f"{base}_full_voxel_loss{ext}"

    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ 计算完成！Loss统计: min={df['voxel_mse_loss'].min():.2e}, "
          f"max={df['voxel_mse_loss'].max():.2e}, mean={df['voxel_mse_loss'].mean():.2e}")
    print(f"💾 结果已保存至: {output_csv_path}")
    return df


if __name__ == "__main__":
    nifti_file = r"data/voxel_data/uniformed_liver_6.nii.gz"
    pred_csv = "./data/pose_predictions.csv"  # 包含 pre_*/tru_* 列

    compute_full_voxel_pose_loss(
        nifti_path=nifti_file,
        csv_path=pred_csv,
        intensity_threshold=0.0,  # 肝脏CT通常HU>0，设为0可滤除背景/空气
        point_batch_size=250_000,  # 根据显存调整 (25万点约占用 ~12MB 显存/样本批)
        sample_batch_size=32,  # 并行计算32个样本/步
        device="cuda"
    )