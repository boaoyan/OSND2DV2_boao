import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')


def compute_pose_errors(csv_path,
                        rot_axes=['x', 'y', 'z'],  # 分析顺序: x→y→z
                        trans_axes=['x', 'y', 'z'],
                        abs_error=True):
    """
    从CSV计算姿态预测误差（支持任意轴顺序）

    Returns:
    --------
    dict : {
        'rot_errors': {axis: error_array},   # 旋转误差数组
        'trans_errors': {axis: error_array}, # 平移误差数组
        'rot_stats': {axis: {'mean', 'std', 'max', 'min'}},
        'trans_stats': {...}
    }
    """
    df = pd.read_csv(csv_path)

    results = {'rot_errors': {}, 'trans_errors': {}, 'rot_stats': {}, 'trans_stats': {}}

    # === 计算旋转误差 ===
    for axis in rot_axes:
        pre_col = f'pre_rota_{axis}'
        tru_col = f'tru_rota_{axis}'

        if pre_col not in df.columns or tru_col not in df.columns:
            print(f"⚠️  缺少列: {pre_col} 或 {tru_col}")
            continue

        diff = df[pre_col] - df[tru_col]
        if abs_error:
            diff = np.abs(diff)

        errors = diff.dropna().values
        results['rot_errors'][axis] = errors
        results['rot_stats'][axis] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'max': np.max(errors),
            'min': np.min(errors),
            'count': len(errors)
        }

    # === 计算平移误差 ===
    for axis in trans_axes:
        pre_col = f'pre_trans_{axis}'
        tru_col = f'tru_trans_{axis}'

        if pre_col not in df.columns or tru_col not in df.columns:
            print(f"⚠️  缺少列: {pre_col} 或 {tru_col}")
            continue

        diff = df[pre_col] - df[tru_col]
        if abs_error:
            diff = np.abs(diff)

        errors = diff.dropna().values
        results['trans_errors'][axis] = errors
        results['trans_stats'][axis] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'max': np.max(errors),
            'min': np.min(errors),
            'count': len(errors)
        }

    return results


def plot_pose_error_analysis(results, model_name,
                             rot_labels=None, trans_labels=None,
                             save_path=None, figsize=(10, 5)):
    """
    绘制姿态误差分析图：旋转误差 + 平移误差 双柱状图
    """
    # 默认标签 (LaTeX格式)
    if rot_labels is None:
        rot_labels = [r'$\Delta \alpha_x(°)$', r'$\Delta \alpha_y(°)$', r'$\Delta \alpha_z(°)$']
    if trans_labels is None:
        trans_labels = [r'$\Delta t_x(mm)$', r'$\Delta t_y(mm)$', r'$\Delta t_z(mm)$']

    # 提取统计值（按x,y,z顺序）
    axes_order = ['x', 'y', 'z']
    rot_means = [results['rot_stats'][ax]['mean'] for ax in axes_order if ax in results['rot_stats']]
    rot_stds = [results['rot_stats'][ax]['std'] for ax in axes_order if ax in results['rot_stats']]
    trans_means = [results['trans_stats'][ax]['mean'] for ax in axes_order if ax in results['trans_stats']]
    trans_stds = [results['trans_stats'][ax]['std'] for ax in axes_order if ax in results['trans_stats']]

    # 创建画布
    fig, (ax_rot, ax_trans) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Pose Estimation Error Analysis - {model_name}', fontsize=14, y=0.98, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 自定义配色

    # === 左图：旋转误差（带误差棒）===
    x_rot = np.arange(len(rot_means))
    bars_rot = ax_rot.bar(x_rot, rot_means, yerr=rot_stds, capsize=5,
                          color=colors[:len(rot_means)], edgecolor='black', linewidth=0.8)

    # 添加数值标签
    for bar, mean_val, std_val in zip(bars_rot, rot_means, rot_stds):
        height = bar.get_height()
        ax_rot.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f'{mean_val:.4f}±{std_val:.4f}',
                    ha='center', va='bottom', fontsize=9, rotation=0)

    ax_rot.set_xticks(x_rot)
    ax_rot.set_xticklabels(rot_labels[:len(rot_means)], fontsize=11)
    ax_rot.set_ylabel('Mean Absolute Error (°)', fontsize=11)
    ax_rot.grid(axis='y', linestyle='--', alpha=0.5)
    ax_rot.set_axisbelow(True)
    ax_rot.set_ylim(0, max(rot_means) * 1.3 if rot_means else 1)

    # === 右图：平移误差（带误差棒）===
    x_trans = np.arange(len(trans_means))
    bars_trans = ax_trans.bar(x_trans, trans_means, yerr=trans_stds, capsize=5,
                              color=colors[:len(trans_means)], edgecolor='black', linewidth=0.8)

    for bar, mean_val, std_val in zip(bars_trans, trans_means, trans_stds):
        height = bar.get_height()
        ax_trans.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                      f'{mean_val:.4f}±{std_val:.4f}',
                      ha='center', va='bottom', fontsize=9, rotation=0)

    ax_trans.set_xticks(x_trans)
    ax_trans.set_xticklabels(trans_labels[:len(trans_means)], fontsize=11)
    ax_trans.grid(axis='y', linestyle='--', alpha=0.5)
    ax_trans.set_axisbelow(True)
    ax_trans.set_ylim(0, max(trans_means) * 1.3 if trans_means else 1)

    # === 图例 ===
    legend_elements = [
        Patch(color=colors[0], label='X-axis', edgecolor='black'),
        Patch(color=colors[1], label='Y-axis', edgecolor='black'),
        Patch(color=colors[2], label='Z-axis', edgecolor='black')
    ]
    ax_rot.legend(handles=legend_elements, title='Axis', fontsize=9, loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"✓ 图片已保存: {save_path}")

    plt.show()
    return fig




# === 1. 配置参数 ===
csv_file = '../data/mix_model_output/model_output_rlat_mix2_model.csv'
model_name = 'RLAT-mix2'

# === 2. 计算误差 ===
print("🔄 Computing pose errors...")
results = compute_pose_errors(csv_file, rot_axes=['x','y','z'], trans_axes=['x','y','z'])

# === 3. 打印统计结果 ===
# print_error_summary(results, model_name)

# === 4. 绘制对比图 ===
print("📈 Generating visualization...")
fig = plot_pose_error_analysis(
    results,
    model_name=model_name,
    rot_labels=[r'$\Delta \alpha_x(°)$', r'$\Delta \alpha_y(°)$', r'$\Delta \alpha_z(°)$'],
    trans_labels=[r'$\Delta t_x(mm)$', r'$\Delta t_y(mm)$', r'$\Delta t_z(mm)$'],
    save_path='pose_error_analysis.png',  # 可选：保存高清图片
    figsize=(10, 5)
)
