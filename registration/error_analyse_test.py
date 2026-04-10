import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import warnings

warnings.filterwarnings('ignore')


def compute_mean_errors(file_path):
    """
    计算单个 CSV 文件的 6 DoF 平均 L1 误差
    🔥 修改：支持单个 CSV 文件路径（不是文件夹）
    🔥 适配：长格式列名 pre_rota_z, pre_rota_x, pre_rota_y (ZXY 顺序)
    ⚠️  保留：硬编码顺序风险、无标准差、无返回值
    """
    # === 1. 轴键名（保持硬编码，保留风险）===
    rot_keys = ['rx', 'ry', 'rz']  # 🔒 硬编码顺序
    trans_keys = ['tx', 'ty', 'tz']

    # === 2. CSV 实际列名映射（ZXY 顺序）===
    # CSV 列：pre_rota_z, pre_rota_x, pre_rota_y (第 1,2,3 个旋转列)
    # 映射：rot_keys[0]='rx' → 实际读取第 1 列 (z)
    #      rot_keys[1]='ry' → 实际读取第 2 列 (x)
    #      rot_keys[2]='rz' → 实际读取第 3 列 (y)
    rot_csv_order = ['z', 'x', 'y']  # 🔥 CSV 中的实际顺序
    trans_csv_order = ['z', 'x', 'y']

    # === 3. 列名前缀（长格式）===
    rot_prefix = 'rota_'
    trans_prefix = 'trans_'

    all_errors = {k: [] for k in rot_keys + trans_keys}

    # === 4. 🔥 读取单个 CSV 文件（不是文件夹）===
    try:
        df = pd.read_csv(file_path)

        # === 5. 计算旋转误差（按 CSV 的 ZXY 顺序读取）===
        for i, key in enumerate(rot_keys):
            csv_axis = rot_csv_order[i]  # 'rx'→'z', 'ry'→'x', 'rz'→'y'
            pre_col = f'pre_{rot_prefix}{csv_axis}'
            tru_col = f'tru_{rot_prefix}{csv_axis}'

            if pre_col in df.columns and tru_col in df.columns:
                err = np.abs(df[pre_col].values - df[tru_col].values)
                all_errors[key].extend(err)
            else:
                print(f"⚠️  缺少列：{pre_col} 或 {tru_col}")

        # === 6. 计算平移误差（按 CSV 的 ZXY 顺序读取）===
        for i, key in enumerate(trans_keys):
            csv_axis = trans_csv_order[i]  # 'tx'→'z', 'ty'→'x', 'tz'→'y'
            pre_col = f'pre_{trans_prefix}{csv_axis}'
            tru_col = f'tru_{trans_prefix}{csv_axis}'

            if pre_col in df.columns and tru_col in df.columns:
                err = np.abs(df[pre_col].values - df[tru_col].values)
                all_errors[key].extend(err)
            else:
                print(f"⚠️  缺少列：{pre_col} 或 {tru_col}")

        print(f"✓ 成功读取：{file_path} ({len(df)} 行)")

    except Exception as e:
        print(f"❌ 读取失败 {file_path}: {str(e)}")
        return {k: np.nan for k in all_errors}

    # === 7. 仅返回均值（保留原逻辑，不添加 std）===
    means = {k: np.mean(all_errors[k]) if all_errors[k] else np.nan for k in all_errors}
    return means


def plot_multi_model_comparison(files, model_names):
    """
    多模型误差对比图
    🔥 修改：输入是 CSV 文件列表（不是文件夹列表）
    ⚠️  保留：硬编码标签、无误差棒、无返回值
    """
    # === 1. 硬编码顺序和标签（保留风险）===
    rot_order = ['rx', 'ry', 'rz']
    rot_labels = [r'$\Delta \alpha(°)$', r'$\Delta \beta(°)$', r'$\Delta \gamma(°)$']

    trans_order = ['tx', 'ty', 'tz']
    trans_labels = [r'$\Delta t_x(mm)$', r'$\Delta t_y(mm)$', r'$\Delta t_z(mm)$']

    # === 2. 收集所有模型的统计结果 ===
    all_means = []
    for file in files:
        err_dict = compute_mean_errors(file)  # 🔥 传入文件路径
        ordered_vals = [err_dict[k] for k in rot_order + trans_order]
        all_means.append(ordered_vals)

    all_means = np.array(all_means)
    rot_means = all_means[:, :3]
    trans_means = all_means[:, 3:]

    # === 3. 绘图配置 ===
    n_models = len(model_names)
    width = 0.8 / n_models
    colors = plt.cm.tab10.colors[:n_models]

    fig, (ax_rot, ax_trans) = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Pose Estimation Error Comparison (Ours)', fontsize=14, y=0.96)

    # === 4. 左图：旋转误差 ===
    x_rot = np.arange(len(rot_order))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax_rot.bar(x_rot + offset, rot_means[i], width, label="", color=color)
        for bar, val in zip(bars, rot_means[i]):
            if not np.isnan(val):
                ax_rot.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + rot_means.max() * 0.01,
                            f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation_mode='anchor')

    ax_rot.set_xticks(x_rot)
    ax_rot.set_xticklabels(rot_labels, fontsize=12)
    ax_rot.set_ylabel('Mean Absolute Error', fontsize=12)
    ax_rot.grid(axis='y', linestyle='--', alpha=0.6)

    # === 5. 右图：平移误差 ===
    x_trans = np.arange(len(trans_order))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax_trans.bar(x_trans + offset, trans_means[i], width, label="", color=color)
        for bar, val in zip(bars, trans_means[i]):
            if not np.isnan(val):
                ax_trans.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + trans_means.max() * 0.01,
                              f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation_mode='anchor')

    ax_trans.set_xticks(x_trans)
    ax_trans.set_xticklabels(trans_labels, fontsize=12)
    ax_trans.grid(axis='y', linestyle='--', alpha=0.6)

    # === 6. 图例 ===
    legend_handles = [Patch(color=c, label=n) for c, n in zip(colors, model_names)]

    fig.legend(
        handles=legend_handles,
        title="Loss Type",
        loc='upper center',
        bbox_to_anchor=(0.5, 0.92),
        ncol=n_models,
        fontsize=10,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    plt.show()

    # === 7. 无返回值（保留原逻辑）===
    return None

# === 配置 CSV 文件路径（直接传入文件，不是文件夹）===
files = [
    "../data/mix_model_output/model_output_rlat_mix_model.csv",      # 🔥 文件路径
    "../data/mix_model_output/model_output_rlat_rlat_model.csv",       # 🔥 文件路径
]
model_names = ["Mix Model", "RLAT Model"]

# === 直接调用 ===
plot_multi_model_comparison(files, model_names)