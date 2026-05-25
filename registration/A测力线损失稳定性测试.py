import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# 直接使用确定的字体名称（避免列表回退）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 10/11 默认
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 方差计算函数 =================
def compute_variance_stats(data, col_name, confidence=0.95):
    """计算单列误差的高斯方差统计量"""
    values = data[col_name].dropna().values
    n = len(values)
    if n < 2:
        return None

    mean = np.mean(values)
    variance = np.var(values, ddof=1)  # ✅ 无偏估计
    std = np.sqrt(variance)
    var_se = std * np.sqrt(2 / (n - 1))

    alpha = 1 - confidence
    chi2_lower = chi2.ppf(1 - alpha / 2, df=n - 1)
    chi2_upper = chi2.ppf(alpha / 2, df=n - 1)
    ci_var_lower = (n - 1) * variance / chi2_lower
    ci_var_upper = (n - 1) * variance / chi2_upper
    ci_std_lower = np.sqrt(ci_var_lower)
    ci_std_upper = np.sqrt(ci_var_upper)
    cv = (std / mean * 100) if mean != 0 else np.nan

    return {
        'col': col_name, 'n': n, 'mean': mean, 'variance': variance, 'std': std,
        'var_se': var_se, 'ci_var_95': (ci_var_lower, ci_var_upper),
        'ci_std_95': (ci_std_lower, ci_std_upper), 'cv': cv,
        'min': np.min(values), 'max': np.max(values), 'median': np.median(values)
    }


# ================= 2. 方差可视化：多文件折线图（波形图） =================
def plot_variance_lines(file_stats, error_cols, save_path=None):
    """
    绘制多文件方差对比折线图
    X轴: 文件序号/文件名 (N个点)
    Y轴: 方差值 (σ²)
    四条曲线: 每种误差类型一条波形
    """
    n_files = len(file_stats)
    file_names = list(file_stats.keys())
    x_pos = np.arange(n_files)  # [0, 1, 2, ..., N-1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # 为每种误差类型绘制一条曲线
    for col_key, col_info in error_cols.items():
        y_values = []
        for fname in file_names:
            stats_list = file_stats[fname]
            s = next((item for item in stats_list if item['col'] == col_key), None)
            y_values.append(s['variance'] if s is not None else np.nan)

        # 绘制折线 + 数据点标记
        ax.plot(x_pos, y_values,
                label=col_info['label'],
                color=col_info['color'],
                marker='o', markersize=6, markerfacecolor='white',
                markeredgecolor=col_info['color'], markeredgewidth=1.5,
                linewidth=2, alpha=0.9)

        # 添加数值标签（可选：数据点上方显示方差值）
        for i, val in enumerate(y_values):
            if not np.isnan(val):
                ax.text(i, val + np.max(y_values) * 0.02, f'{val:.4f}',
                        ha='center', va='bottom', fontsize=9, fontweight='normal',
                        color=col_info['color'], bbox=dict(boxstyle='round,pad=0.2',
                                                           facecolor='white', edgecolor=col_info['color'], alpha=0.6))

    # ================= 坐标轴与标题配置 =================
    ax.set_xlabel('文件序号 / 数据集', fontsize=11, fontweight='bold')
    ax.set_ylabel('Variance (σ²)', fontsize=11, fontweight='bold')
    ax.set_title('模型误差方差变化波形图 (σ²)', fontsize=14, fontweight='bold', pad=15)

    # X轴刻度：显示文件名（短名称）或序号
    if n_files <= 10:
        # 文件少时显示完整文件名
        short_names = [name[:15] + '...' if len(name) > 15 else name for name in file_names]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    else:
        # 文件多时只显示序号，避免重叠
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'#{i + 1}' for i in x_pos], fontsize=10)

    # 图例、网格、布局
    ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=False)
    ax.grid(True, axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=10)

    # Y轴对数刻度（可选：方差跨度大时启用）
    # ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.show()
    return fig


# ================= 3. 生成方差统计报告 =================
def generate_variance_report(stats_list, error_cols):
    """生成文本格式的方差统计摘要"""
    report = ["📋 高斯方差统计报告", "=" * 70]
    for s in stats_list:
        if s is None:
            continue
        col = s['col']
        info = error_cols[col]
        report.append(f"\n🔹 {info['label']} [{col}]")
        report.append(f"   ├─ 样本量: {s['n']}")
        report.append(f"   ├─ 均值: {s['mean']:.4f} {info['unit']}")
        report.append(f"   ├─ 方差 (σ²): {s['variance']:.6f} {info['unit']}²")
        report.append(f"   ├─ 标准差 (σ): {s['std']:.4f} {info['unit']}")
        report.append(f"   ├─ 方差标准误差: {s['var_se']:.6f}")
        report.append(f"   ├─ 95% CI (方差): [{s['ci_var_95'][0]:.6f}, {s['ci_var_95'][1]:.6f}]")
        report.append(f"   ├─ 95% CI (标准差): [{s['ci_std_95'][0]:.4f}, {s['ci_std_95'][1]:.4f}] {info['unit']}")
        report.append(f"   ├─ 变异系数 (CV): {s['cv']:.1f}%")
        report.append(f"   └─ 数据范围: [{s['min']:.4f}, {s['max']:.4f}] {info['unit']}")
    return "\n".join(report)


# ================= 辅助函数与度量类 =================
def extract_pose(df, prefix):
    rot = df[[f'{prefix}_rota_z', f'{prefix}_rota_x', f'{prefix}_rota_y']].values.astype(np.float32)
    trans = df[[f'{prefix}_trans_x', f'{prefix}_trans_y', f'{prefix}_trans_z']].values.astype(np.float32)
    return rot, trans


def euler_angles_to_rotation_matrix_zxy(euler_angles_deg):
    theta = np.deg2rad(euler_angles_deg)
    cz, sz = np.cos(theta[:, 0]), np.sin(theta[:, 0])
    cx, sx = np.cos(theta[:, 1]), np.sin(theta[:, 1])
    cy, sy = np.cos(theta[:, 2]), np.sin(theta[:, 2])
    R = np.zeros((euler_angles_deg.shape[0], 3, 3), dtype=np.float64)
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


def so3_log_map_np(R):
    tr = np.trace(R, axis1=1, axis2=2)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    denom = np.where(np.abs(sin_theta) > 1e-8, 2.0 * sin_theta, 2.0)
    scale = theta / denom
    log_vec = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1) * scale[:,
                                                                                                              None]
    return log_vec


class DoubleGeodesicSE3:
    def __init__(self, sdd: float, eps: float = 1e-6):
        self.sdr = sdd / 2.0
        self.eps = eps

    def compute(self, pred_R, pred_t, gt_R, gt_t):
        R_rel = np.matmul(np.transpose(pred_R, (0, 2, 1)), gt_R)
        log_vec = so3_log_map_np(R_rel)
        rot_err = self.sdr * np.linalg.norm(log_vec, axis=1)
        xyz_err = np.linalg.norm(pred_t - gt_t, axis=1)
        combined_err = np.sqrt(rot_err ** 2 + xyz_err ** 2 + self.eps)
        return rot_err, xyz_err, combined_err


# ================= 主程序 =================
if __name__ == '__main__':
    # 🔧 支持任意N个CSV文件，直接在此列表中添加路径即可
    csv_paths = [
        "../data/uliver3_data_B/1000_copy1/1000_copy1_pa_mix1_model.csv",
        "../data/uliver3_data_B/1000_copy2/1000_copy2_pa_mix1_model.csv",
        "../data/uliver3_data_B/1000_copy3/1000_copy3_pa_mix1_model.csv",  # 可扩展
        "../data/uliver3_data_B/1000_copy4/1000_copy4_pa_mix1_model.csv",
        "../data/uliver3_data_B/3000_copy1/3000_copy1_pa_mix1_model.csv",
        "../data/uliver3_data_B/3000_copy2/3000_copy2_pa_mix1_model.csv",
        "../data/uliver3_data_B/3000_copy3/3000_copy3_pa_mix1_model.csv",  # 可扩展
        "../data/uliver3_data_B/3000_copy4/3000_copy4_pa_mix1_model.csv",
        "../data/uliver3_data_B/5000_copy1/5000_copy1_pa_mix1_model.csv",
        "../data/uliver3_data_B/5000_copy2/5000_copy2_pa_mix1_model.csv",
        "../data/uliver3_data_B/5000_copy3/5000_copy3_pa_mix1_model.csv",  # 可扩展
        "../data/uliver3_data_B/5000_copy4/5000_copy4_pa_mix1_model.csv"
    ]

    ERROR_COLS = {
        'rot_error_mm': {'label': '旋转误差 (mm)', 'unit': 'mm', 'color': '#4E79A7'},
        'trans_error_mm': {'label': '平移误差 (mm)', 'unit': 'mm', 'color': '#F28E2B'},
        'combined_error_mm': {'label': '综合误差 (mm)', 'unit': 'mm', 'color': '#E15759'},
        'rot_error_deg': {'label': '旋转误差 (°)', 'unit': '°', 'color': '#76B7B2'}
    }
    SDD = 800.0
    metric = DoubleGeodesicSE3(sdd=SDD)

    file_stats = {}  # 存储每个文件的统计结果 {filename: [stats1, stats2, ...]}

    for path in csv_paths:
        if not os.path.exists(path):
            print(f"⚠️ 文件未找到，跳过: {path}")
            continue

        print(f"📊 正在处理: {os.path.basename(path)}")
        df = pd.read_csv(path)

        pre_rot, pre_trans = extract_pose(df, 'pre')
        tru_rot, tru_trans = extract_pose(df, 'tru')
        pre_R = euler_angles_to_rotation_matrix_zxy(pre_rot)
        tru_R = euler_angles_to_rotation_matrix_zxy(tru_rot)

        rot_err, xyz_err, combined_err = metric.compute(pre_R, pre_trans, tru_R, tru_trans)

        results = df.copy()
        results['rot_error_mm'] = rot_err
        results['trans_error_mm'] = xyz_err
        results['combined_error_mm'] = combined_err
        results['rot_error_deg'] = (rot_err / (SDD / 2.0)) * (180.0 / np.pi)

        # 计算该文件所有误差列的统计量
        stats_list = [compute_variance_stats(results, col) for col in ERROR_COLS.keys()]
        file_stats[os.path.basename(path)] = stats_list

        # 打印单个文件的报告
        print(generate_variance_report(stats_list, ERROR_COLS))
        print("-" * 70)

    # 绘制多曲线波形对比图
    if file_stats:
        print(f"✅ 共处理 {len(file_stats)} 个文件，正在生成波形对比图...")
        plot_variance_lines(file_stats, ERROR_COLS, save_path="variance_waveform.png")
    else:
        print("❌ 未读取到有效数据文件。")