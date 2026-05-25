import os
import glob
import csv
import nibabel as nib
import numpy as np


def compute_intensity_stats(data, mask=None):
    """计算体素强度统计特征（仅处理强度数据）"""
    if mask is not None:
        data = data[mask]
    else:
        data = data.flatten()

    data = data[np.isfinite(data)]
    if len(data) == 0:
        return None

    mean_val = np.mean(data)
    std_val = np.std(data)

    return {
        'count': len(data),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(mean_val),
        'std': float(std_val),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'dynamic_range': float(np.max(data) - np.min(data)),
        'cv': float(std_val / (mean_val + 1e-10)),  # 变异系数
        'skewness': float(((data - mean_val) ** 3).mean() / (std_val ** 3 + 1e-10)),
        'kurtosis': float(((data - mean_val) ** 4).mean() / (std_val ** 4 + 1e-10) - 3)  # 超额峰度
    }


def analyze_folder_volumes(folder_path, mask_path=None, save_csv=True):
    """批量分析文件夹中所有 NIfTI 文件的尺寸、间隔与强度统计"""
    # 1. 查找所有 NIfTI 文件
    patterns = [os.path.join(folder_path, '*.nii'), os.path.join(folder_path, '*.nii.gz')]
    nii_files = sorted([f for p in patterns for f in glob.glob(p)])

    if not nii_files:
        print("❌ 未找到任何 NIfTI 文件 (.nii / .nii.gz)")
        return []

    # 2. 加载掩码（如提供）
    mask_data = None
    if mask_path and os.path.exists(mask_path):
        print(f"🎭 加载掩码: {mask_path}")
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)

    results = []
    print(f"📂 开始扫描文件夹: {folder_path} (共 {len(nii_files)} 个文件)\n" + "-" * 50)

    # 3. 逐文件处理
    for filepath in nii_files:
        filename = os.path.basename(filepath)
        try:
            img = nib.load(filepath)
            # 提取空间元数据（兼容4D数据，仅取前3维）
            shape = img.shape[:3]
            spacing = img.header.get_zooms()[:3]
            volume_data = img.get_fdata()

            # 计算强度统计
            stats = compute_intensity_stats(volume_data, mask=mask_data)
            if stats is None:
                print(f"⚠️ 跳过 {filename}: 无有效体素数据")
                continue

            # 合并元数据与统计结果
            record = {
                'filename': filename,
                'dim_x': shape[0], 'dim_y': shape[1], 'dim_z': shape[2],
                'spacing_x_mm': spacing[0], 'spacing_y_mm': spacing[1], 'spacing_z_mm': spacing[2],
                'voxel_volume_mm3': float(np.prod(spacing)),
                **stats
            }
            results.append(record)

            # 终端打印精简报告
            print(f"✅ {filename}")
            print(f"   📐 尺寸: {shape} | 间隔: {spacing} mm | 单 voxel 体积: {record['voxel_volume_mm3']:.4f} mm³")
            print(
                f"   📊 强度: μ={stats['mean']:.4f} ± σ={stats['std']:.4f} | 范围=[{stats['min']:.2f}, {stats['max']:.2f}] | 有效体素: {stats['count']:,}")
            print()

        except Exception as e:
            print(f"❌ 处理 {filename} 失败: {e}\n")
            continue

    # 4. 导出 CSV 汇总
    if save_csv and results:
        csv_path = os.path.join(folder_path, 'volume_stats_summary.csv')
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"💾 批量统计结果已保存至: {csv_path}")

    return results


if __name__ == '__main__':
    # 配置路径
    FOLDER_PATH = '../data/voxel_data/'  # 替换为你的文件夹路径
    # MASK_PATH = '../data/voxel_data/liver_mask.nii.gz'  # 可选：统一掩码路径

    # 执行批量分析
    stats_list = analyze_folder_volumes(
        folder_path=FOLDER_PATH,
        mask_path=None,
        save_csv=True
    )

    if stats_list:
        print(f"\n🎉 成功处理 {len(stats_list)} 个体素文件")