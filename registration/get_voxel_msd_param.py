
import numpy as np
import nibabel as nib
import json
import os
from pathlib import Path


def compute_msd_stats_analytical(shape, affine):
    """
    基于体素形状与仿射矩阵，O(1) 精确计算物理空间(mm)下的坐标统计量

    数学推导:
      索引空间均值: μ_idx = (S-1)/2
      索引空间方差: σ²_idx = (S²-1)/12
      物理空间均值: μ_p = R_aff @ μ_idx + t_aff
      物理空间协方差: Σ_p = R_aff @ diag(σ²_idx) @ R_aff^T
    """
    S = np.array(shape[:3], dtype=np.float64)
    R_aff = affine[:3, :3].astype(np.float64)
    t_aff = affine[:3, 3].astype(np.float64)

    # 1. 索引空间统计量
    mu_idx = (S - 1.0) / 2.0
    var_idx = (S ** 2 - 1.0) / 12.0

    # 2. 物理空间映射 (仿射变换线性性质)
    mu_p = R_aff @ mu_idx + t_aff
    Sigma_idx = np.diag(var_idx)
    Sigma_p = R_aff @ Sigma_idx @ R_aff.T

    return mu_p, Sigma_p


def extract_volume_msd_params(vol_path, save_json=True, verbose=True):
    """
    读取体素文件并提取闭式解 MSD 所需参数

    参数:
        vol_path: .nii.gz 或 .npy 路径
        save_json: 是否保存为 JSON 配置文件
        verbose: 是否打印详细信息

    返回:
        dict: {'mu_p': ndarray(3,), 'Sigma_p': ndarray(3,3), 'shape', 'spacing', 'N'}
    """
    path = Path(vol_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 文件不存在: {vol_path}")

    # 1. 加载文件头信息
    if path.suffix in ['.nii', '.gz'] or '.nii.gz' in path.name:
        img = nib.load(str(path))
        shape = img.shape
        affine = img.affine
        spacing = img.header.get_zooms()[:3]
    elif path.suffix == '.npy':
        # .npy 默认假设各向同性 1mm 且无旋转
        shape = np.load(str(path), mmap_mode='r').shape
        affine = np.diag(list(spacing := (1.0, 1.0, 1.0)) + [1.0])
    else:
        raise ValueError("❌ 仅支持 .nii.gz 或 .npy 格式")

    # 2. O(1) 精确计算
    mu_p, Sigma_p = compute_msd_stats_analytical(shape, affine)
    N = int(np.prod(shape))

    # 3. 封装结果
    result = {
        'shape': shape[:3],
        'affine': affine.tolist(),
        'spacing': tuple(spacing),
        'N': N,
        'mu_p': mu_p.tolist(),
        'Sigma_p': Sigma_p.tolist(),
        'unit': 'mm',
        'source': str(path)
    }

    # 4. 输出
    if verbose:
        print(f"📦 体素参数提取完成: {path.name}")
        print(f"   • 形状: {result['shape']} | 间隔: {result['spacing']} mm")
        print(f"   • 空间仿射变换矩阵: \n{result['affine']}")
        print(f"   • 体素总数: {N:,}")
        print(f"   • 几何中心 μ_p: {mu_p}")
        print(f"   • 协方差矩阵 Σ_p:\n{Sigma_p}")
        print(f"   • 坐标系: 物理空间 (mm) | 原点: NIfTI Affine 定义")

    # if save_json:
    #     json_path = path.with_suffix('.msd_stats.json')
    #     with open(json_path, 'w') as f:
    #         json.dump(result, f, indent=2)
    #     if verbose: print(f"💾 已保存至: {json_path}")

    return result


# ================= 执行入口 =================
if __name__ == "__main__":
    # 示例：替换为你的实际路径
    VOL_FILE = "../data/voxel_data/uniformed_liver_3.nii.gz"
    extract_volume_msd_params(VOL_FILE, save_json=True, verbose=True)


