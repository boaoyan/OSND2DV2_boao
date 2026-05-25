import torch
import numpy as np
import nibabel as nib
import os
import pandas as pd
from tqdm import tqdm
import sys


# ================= 1. 核心数学工具 =================
def euler_zxy_to_rot_batch(angles_deg: torch.Tensor) -> torch.Tensor:
    rad = angles_deg * (torch.pi / 180.0)
    z, x, y = rad[:, 0], rad[:, 1], rad[:, 2]
    c, s = torch.cos, torch.sin
    zeros, ones = torch.zeros_like(z), torch.ones_like(z)

    Rz = torch.stack([torch.stack([c(z), -s(z), zeros], dim=1),
                      torch.stack([s(z), c(z), zeros], dim=1),
                      torch.stack([zeros, zeros, ones], dim=1)], dim=1)
    Rx = torch.stack([torch.stack([ones, zeros, zeros], dim=1),
                      torch.stack([zeros, c(x), -s(x)], dim=1),
                      torch.stack([zeros, s(x), c(x)], dim=1)], dim=1)
    Ry = torch.stack([torch.stack([c(y), zeros, s(y)], dim=1),
                      torch.stack([zeros, ones, zeros], dim=1),
                      torch.stack([-s(y), zeros, c(y)], dim=1)], dim=1)
    return Ry @ Rx @ Rz


def compute_true_loss_fast(pts_phys: torch.Tensor, pose_errors: torch.Tensor,
                           device: str = "cuda", chunk_size: int = 8192) -> torch.Tensor:
    pts_phys = pts_phys.to(device)
    pose_errors = pose_errors.to(device)

    d_rot, d_trans = pose_errors[:, :3], pose_errors[:, 3:]
    R_pred = euler_zxy_to_rot_batch(d_rot)
    R_diff = torch.eye(3, device=device).unsqueeze(0).expand(len(pose_errors), -1, -1) - R_pred
    t_diff = -d_trans

    B = len(pose_errors)
    losses = torch.zeros(B, device="cpu")
    N = pts_phys.size(0)

    for i in range(0, B, chunk_size):
        i_end = min(i + chunk_size, B)
        R_b, t_b = R_diff[i:i_end], t_diff[i:i_end]
        pts_b = pts_phys.T.unsqueeze(0).expand(R_b.size(0), -1, -1)
        disp = torch.bmm(R_b, pts_b) + t_b.unsqueeze(-1)
        losses[i:i_end] = (disp ** 2).sum(dim=[1, 2]).cpu() / (N * 3.0)
        torch.cuda.empty_cache()
    return losses


# ================= 2. 单文件加载与预处理 =================
def load_and_preprocess_voxel(fp: str, max_voxels: int = 50000, device: str = "cuda"):
    ext = os.path.splitext(fp)[-1]
    if ext == '.gz':
        ext = os.path.splitext(fp[:-3])[-1]  # 处理 .nii.gz

    if ext in ['.nii', '.gz']:
        img = nib.load(fp)
        data = img.get_fdata()
        affine = torch.tensor(img.affine, dtype=torch.float32, device=device)
        valid = data > 0
        idx = np.argwhere(valid).astype(np.float32)[:, [2, 1, 0]]
        hom = np.concatenate([idx, np.ones((len(idx), 1), dtype=np.float32)], axis=1)
        pts = torch.tensor(hom, device=device) @ affine.T
        pts = pts[:, :3]
    elif ext in ['.pt', '.pth']:
        pts = torch.load(fp, map_location=device)
        if pts.dim() == 2 and pts.shape[1] != 3: pts = pts.T
    elif ext == '.npy':
        pts = torch.tensor(np.load(fp), dtype=torch.float32, device=device)
        if pts.dim() == 2 and pts.shape[1] != 3: pts = pts.T
    else:
        raise ValueError(f"❌ 不支持的文件格式: {ext}")

    # 重要性下采样
    if len(pts) > max_voxels:
        center = pts.mean(dim=0)
        dists = torch.norm(pts - center, dim=1)
        probs = (dists ** 2) / (dists ** 2).sum()
        pts = pts[torch.multinomial(probs, max_voxels, replacement=False)]

    # 🔥 修复版：提取轻量几何特征 [16]
    # 所有计算均在 pts 所在设备(device)进行，避免 torch.cat 设备冲突
    cov = torch.cov(pts.T)
    geo = torch.cat([
        pts.mean(dim=0),  # [3] 质心
        pts.std(dim=0),  # [3] 散布
        (pts ** 2).mean(dim=0),  # [3] 二阶矩
        pts.max(dim=0)[0] - pts.min(dim=0)[0],  # [3] 跨度
        torch.diag(cov),  # [3] 协方差对角
        torch.trace(cov).unsqueeze(0)  # [1] ✅ 修复：替代 torch.tensor()，保持设备一致
    ])

    # 计算完成后统一移回 CPU 便于保存
    return pts.cpu(), geo.cpu()


# ================= 3. 主流程：单文件双输出 =================
def generate_single_voxel_dataset(
        pose_errors_csv: str,
        voxel_file_path: str,  # ← 仅接收单个文件路径
        output_dir: str = "data/surrogate_datasets",
        filter_idx: int = 15,
        sample_idx_col: str = "sample_idx",
        max_voxels: int = 50000,
        chunk_size: int = 8192,
        device: str = "cuda"
):
    print(f"🚀 启动单文件代理数据集生成 | 设备: {device}")
    print(f"📂 处理体素文件: {voxel_file_path}")
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ 加载位姿误差CSV
    print(f"📖 加载位姿误差CSV: {pose_errors_csv}")
    df_err = pd.read_csv(pose_errors_csv)
    err_cols = ['tru_rota_z', 'tru_rota_x', 'tru_rota_y', 'tru_trans_x', 'tru_trans_y', 'tru_trans_z']
    pred_cols = ['pre_rota_z', 'pre_rota_x', 'pre_rota_y', 'pre_trans_x', 'pre_trans_y', 'pre_trans_z']
    errors = torch.tensor(df_err[err_cols].values - df_err[pred_cols].values, dtype=torch.float32)
    print(f"   📊 位姿误差形状: {errors.shape}")

    # 构建过滤掩码
    if sample_idx_col in df_err.columns:
        full_mask = torch.ones(len(df_err), dtype=torch.bool)
        train_mask = torch.tensor(df_err[sample_idx_col].values == filter_idx, dtype=torch.bool)
    else:
        print(f"   ⚠️ 未找到 '{sample_idx_col}' 列，默认取前20000行为训练集")
        full_mask = torch.ones(len(df_err), dtype=torch.bool)
        train_mask = torch.zeros(len(df_err), dtype=torch.bool)
        train_mask[:20000] = True
    print(f"   🔍 全量: {full_mask.sum()} | 训练集过滤条件: {train_mask.sum()}")

    # 2️⃣ 加载当前单个体素文件
    pts, geo_stats = load_and_preprocess_voxel(voxel_file_path, max_voxels, device)
    print(f"   📐 体素坐标加载完成: {len(pts)} 个点 | 几何特征已提取")

    # 3️⃣ 计算全量Loss
    print("   ⚡ 计算32万样本Loss...")
    all_losses = compute_true_loss_fast(pts, errors, device, chunk_size)

    # 4️⃣ 保存全量文件
    # 安全提取文件名（兼容 .nii.gz / .pt / .npy）
    basename = os.path.basename(voxel_file_path)
    for ext in ['.nii.gz', '.nii', '.pt', '.pth', '.npy']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    fname = basename

    full_data = {
        "pose_errors": errors[full_mask].cpu(),
        "true_losses": all_losses[full_mask].cpu(),
        "voxel_geo_stats": geo_stats.cpu(),
        "metadata": {"voxel_file": fname, "num_voxels": len(pts)}
    }
    full_path = os.path.join(output_dir, f"{fname}_full.pt")
    torch.save(full_data, full_path)
    print(f"   ✅ 全量已保存: {full_path} ({full_data['pose_errors'].shape[0]:,} 样本)")

    # 5️⃣ 保存训练集文件
    train_data = {
        "pose_errors": errors[train_mask].cpu(),
        "true_losses": all_losses[train_mask].cpu(),
        "voxel_geo_stats": geo_stats.cpu(),
        "metadata": {"voxel_file": fname, "num_voxels": len(pts), "filter_idx": filter_idx}
    }
    train_path = os.path.join(output_dir, f"{fname}_train.pt")
    torch.save(train_data, train_path)
    print(f"   ✅ 训练集已保存: {train_path} ({train_data['pose_errors'].shape[0]:,} 样本)")

    # 清理显存
    del pts, all_losses, full_data, train_data, errors, df_err
    torch.cuda.empty_cache()
    print("   🧹 显存已清理，单文件处理完毕。")


# ================= 运行入口 =================
if __name__ == "__main__":
    # 🔧 单次运行配置（每次只改这一行即可处理不同体素）
    SINGLE_VOXEL_FILE = "data/voxel_data/uniformed_liver_9.nii.gz"  # ← 替换为你的实际单文件路径
    POSE_CSV = "data/loss/noise_physical.csv"
    OUT_DIR = "data/surrogate_datasets"

    generate_single_voxel_dataset(
        pose_errors_csv=POSE_CSV,
        voxel_file_path=SINGLE_VOXEL_FILE,
        output_dir=OUT_DIR,
        filter_idx=13,
        sample_idx_col="sample_idx",
        max_voxels=50000,
        chunk_size=8192,
        device="cuda"
    )
