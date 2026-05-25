import csv
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from registration.math_process.data_fusion import get_base_noise, get_source_transform, get_osz_noise

# 设置中文字体（根据系统调整，避免乱码）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False


# 请确保以下辅助函数已导入：
# extract_pose_named_data, get_base_noise, get_osz_noise,
# get_source_transform, compute_pose_errors_unknown_mean, reg_data_fusion

def extract_pose_named_data(csv_paths,
                            rot_cols=['pre_rota_z', 'pre_rota_x', 'pre_rota_y'],
                            trans_cols=['pre_trans_x', 'pre_trans_y', 'pre_trans_z'],
                            tru_rot_cols=['tru_rota_z', 'tru_rota_x', 'tru_rota_y'],
                            tru_trans_cols=['tru_trans_x', 'tru_trans_y', 'tru_trans_z'],
                            strip_suffix='_model'):
    all_data = {}
    req_cols = rot_cols + trans_cols + tru_rot_cols + tru_trans_cols

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        pt_raw = df['pose_type'].iloc[0]
        mn_raw = df['model_name'].iloc[0]

        mn_clean = str(mn_raw).replace(strip_suffix, '') if strip_suffix else str(mn_raw)
        pt = re.sub(r'[^a-z0-9]', '_', pt_raw.lower().strip()).strip('_')
        mn = re.sub(r'[^a-z0-9]', '_', mn_clean.lower().strip()).strip('_')
        prefix = f"{pt}_{mn}"

        valid_df = df.dropna(subset=req_cols).copy()
        if valid_df.empty:
            continue

        all_data[f"{prefix}_pre_rota"] = valid_df[rot_cols].to_numpy(np.float64)
        all_data[f"{prefix}_pre_trans"] = valid_df[trans_cols].to_numpy(np.float64)
        all_data[f"{prefix}_tru_rota"] = valid_df[tru_rot_cols].to_numpy(np.float64)
        all_data[f"{prefix}_tru_trans"] = valid_df[tru_trans_cols].to_numpy(np.float64)

    return all_data

def compute_pose_errors_unknown_mean(pre_rota, tru_rota, pre_trans, tru_trans,
                                     rot_units='deg', handle_angle_wrap=True):
    """
    姿态误差计算（均值未知情形 → 使用无偏样本方差）

    轴顺序约定：
    - 旋转: [z, x, y] → 索引 0,1,2
    - 平移: [x, y, z] → 索引 0,1,2

    假设误差服从高斯分布，但均值未知（可能存在系统偏差）。
    使用公式：S² = 1/(N-1) * Σ(e_i - ē)²

    Returns
    -------
    dict : {
        'rot': {'x': {'mean', 'var', 'std'}, 'y': {...}, 'z': {...}},
        'trans': {'x': {'mean', 'var', 'std'}, 'y': {...}, 'z': {...}}
    }
    """
    results = {'rot': {}, 'trans': {}}
    ddof = 1  # 均值未知时的自由度修正（N-1）

    # === 旋转误差 ===
    rot_axis_map = {0: 'z', 1: 'x', 2: 'y'}
    for idx, axis in rot_axis_map.items():
        if pre_rota.ndim < 2 or pre_rota.shape[1] <= idx:
            continue
        diff = pre_rota[:, idx] - tru_rota[:, idx]

        # 角度周期性处理
        if handle_angle_wrap and rot_units == 'deg':
            diff = (diff + 180) % 360 - 180

        valid = diff[~np.isnan(diff)]
        n = len(valid)

        if n > 1:
            mean_val = float(np.mean(valid))
            var_val = float(np.var(valid, ddof=ddof))
            results['rot'][axis] = {
                'mean': mean_val,
                'var': var_val
            }
        elif n == 1:
            # 单样本无法计算无偏方差，返回 NaN
            results['rot'][axis] = {'mean': float(valid[0]), 'var': np.nan, 'std': np.nan}

    # === 平移误差 ===
    trans_axes = ['x', 'y', 'z']
    for idx, axis in enumerate(trans_axes):
        if pre_trans.ndim < 2 or pre_trans.shape[1] <= idx:
            continue
        diff = pre_trans[:, idx] - tru_trans[:, idx]
        valid = diff[~np.isnan(diff)]
        n = len(valid)

        if n > 1:
            mean_val = float(np.mean(valid))
            var_val = float(np.var(valid, ddof=ddof))
            results['trans'][axis] = {
                'mean': mean_val,
                'var': var_val
            }
        elif n == 1:
            results['trans'][axis] = {'mean': float(valid[0]), 'var': np.nan, 'std': np.nan}

    return results

def reg_data_fusion(error1, error2,
                    pre_rota1, pre_trans1,
                    pre_rota2, pre_trans2,
                    angle_unit='rad',
                    rot_order=('rz', 'rx', 'ry')):  # 🆕 显式声明旋转列顺序
    """
    6-DoF 位姿加权融合（已修复方差与预测值轴顺序不一致问题）

    Parameters
    ----------
    error1, error2 : dict 或 array
        方差输入。数组顺序固定为: [rx, ry, rz, tx, ty, tz]
    pre_rota1/2 : array-like, shape (N, 3)
        旋转预测值。默认列顺序: ['rz', 'rx', 'ry']
    pre_trans1/2 : array-like, shape (N, 3)
        平移预测值。顺序: [tx, ty, tz]
    angle_unit : str, default 'rad'
        保留参数，实际按输入原始数值计算
    rot_order : tuple, default ('rz', 'rx', 'ry')
        旋转数据的实际列顺序，用于自动对齐方差

    Returns
    -------
    fused_rot : np.ndarray, shape (3,) or (N, 3)
    fused_trans : np.ndarray, shape (3,) or (N, 3)
    """

    # 🔧 内部辅助：解析方差
    def _parse_variance(err_input):
        if isinstance(err_input, dict):
            axes = ['x', 'y', 'z']
            try:
                r_vars = [err_input['rot'][ax]['var'] for ax in axes]
                t_vars = [err_input['trans'][ax]['var'] for ax in axes]
                return np.array(r_vars + t_vars)  # 返回 [rx, ry, rz, tx, ty, tz]
            except KeyError as e:
                raise ValueError(f"误差字典结构缺失键: {e}")
        else:
            return np.asarray(err_input, dtype=float)

    # 统一转为 2D
    var1 = np.atleast_2d(_parse_variance(error1))
    var2 = np.atleast_2d(_parse_variance(error2))
    r1 = np.atleast_2d(np.asarray(pre_rota1, dtype=float))
    t1 = np.atleast_2d(np.asarray(pre_trans1, dtype=float))
    r2 = np.atleast_2d(np.asarray(pre_rota2, dtype=float))
    t2 = np.atleast_2d(np.asarray(pre_trans2, dtype=float))

    # 维度校验
    if var1.shape != var2.shape or var1.shape[1] != 6:
        raise ValueError("Variance must be shape (6,) or (N, 6)")
    if r1.shape != r2.shape or t1.shape != t2.shape:
        raise ValueError("Rotation/Translation shapes must match")
    if r1.shape[1] != 3 or t1.shape[1] != 3:
        raise ValueError("Rot/Trans must be shape (3,) or (N, 3)")

    # 1. 计算基础权重
    total_var = var1 + var2
    total_var = np.where(total_var == 0, 1e-12, total_var)
    w1 = var2 / total_var  # error1 的权重
    w2 = var1 / total_var  # error2 的权重

    # 2. 🆕 关键修复：对齐旋转轴顺序
    # 输入方差顺序: [rx, ry, rz] -> 索引 0,1,2
    # 输入旋转顺序: rot_order (默认 rz, rx, ry) -> 需将方差权重重排匹配
    rot_map = {'rz': 2, 'rx': 0, 'ry': 1}
    rot_idx = [rot_map[axis] for axis in rot_order]  # 默认 [2, 0, 1]

    w1_rot = w1[:, rot_idx]  # 重排为 [rz, rx, ry]
    w2_rot = w2[:, rot_idx]

    w1_trans = w1[:, 3:]  # [tx, ty, tz] 已对齐，无需重排
    w2_trans = w2[:, 3:]

    # 3. 线性融合
    fused_rot = w1_rot * r1 + w2_rot * r2
    fused_trans = w1_trans * t1 + w2_trans * t2

    # 恢复 1D
    if fused_rot.shape[0] == 1:
        return fused_rot.squeeze(), fused_trans.squeeze()
    return fused_rot, fused_trans


def parse_folder_label(folder_name):
    """
    解析文件夹名称，生成简洁横轴标签
    输入: "1000_copy1" → 输出: "1000_1"
    输入: "8000_copy4" → 输出: "8000_4"
    """
    import re
    # 匹配 pattern: 数字_copy数字
    match = re.match(r'(\d+)_copy(\d+)', folder_name.strip())
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    # 若不符合格式，返回原名称（截断前15字符避免过长）
    return folder_name[:15]


def plot_pose_error_waveforms(data_dict, folder_names, save_dir, dpi=400):
    """
    绘制姿态误差方差波形图（大字体高清版 + 文件夹名称横轴标签）

    横轴标签示例: 1000_1, 2000_2, 8000_4 (由文件夹名解析生成)
    """
    os.makedirs(save_dir, exist_ok=True)

    error_labels = {
        'pa_mix1': 'PA-Mix1 (正位)', 'rlat_mix1': 'RLat-Mix1 (侧位)',
        'rlat2pa_mix1': 'RLat→PA-Mix1', 'fusion_mix1': 'Fusion-Mix1',
        'pa_mix2': 'PA-Mix2', 'rlat_mix2': 'RLat-Mix2',
        'rlat2pa_mix2': 'RLat→PA-Mix2', 'fusion_mix2': 'Fusion-Mix2',
        'pa_mix3': 'PA-Mix3', 'rlat_mix3': 'RLat-Mix3',
        'rlat2pa_mix3': 'RLat→PA-Mix3', 'fusion_mix3': 'Fusion-Mix3'
    }

    # 维度配置：用 TeX 数学模式渲染上标，兼容所有字体
    dim_config = [
        ('rot', 'z', r'$^\circ$$^2$', '#E41A1C', False),
        ('rot', 'x', r'$^\circ$$^2$', '#377EB8', False),
        ('rot', 'y', r'$^\circ$$^2$', '#4DAF4A', False),
        ('trans', 'x', r'$mm^2$', '#FF7F00', True),
        ('trans', 'y', r'$mm^2$', '#984EA3', True),
        ('trans', 'z', r'$mm^2$', '#F781BF', True),
    ]

    # === 🎯 单图动态范围配置参数 ===
    RANGE_MARGIN = 0.30
    MIN_ROT_SPAN = 0.003
    MIN_TRANS_SPAN = 0.001
    STABLE_BAND_RATIO = 0.5
    # ==============================

    x_axis = np.arange(len(folder_names))
    # 🔑 核心改动：使用解析后的文件夹名称作为横轴标签
    x_labels = [parse_folder_label(name) for name in folder_names]

    for err_key in data_dict.keys():
        err_list = data_dict[err_key]
        if not err_list:
            print(f"  ⚠️ {err_key} 无有效数据，跳过绘图")
            continue

        # 🔍 1. 【单图独立】计算动态坐标范围
        rot_vals, trans_vals = [], []
        for err_dict in err_list:
            for comp, axis, unit, color, use_right in dim_config:
                val = err_dict.get(comp, {}).get(axis, {}).get('var', np.nan)
                if not np.isnan(val) and val >= 0:
                    if comp == 'rot':
                        rot_vals.append(val)
                    else:
                        trans_vals.append(val)

        def calc_single_lim(values, margin, min_span):
            if not values:
                return (-min_span * 0.2, min_span)
            v_max = np.max(values)
            ylim_max = max(v_max * (1 + margin), min_span)
            ylim_min = -min_span * 0.2
            return (ylim_min, ylim_max)

        rot_ylim = calc_single_lim(rot_vals, RANGE_MARGIN, MIN_ROT_SPAN)
        trans_ylim = calc_single_lim(trans_vals, RANGE_MARGIN, MIN_TRANS_SPAN)

        # 🔍 2. 收集各维度值
        dim_values = {f"{comp}_{axis}": [] for comp, axis, _, _, _ in dim_config}
        for comp, axis, unit, color, use_right in dim_config:
            values = []
            for err_dict in err_list:
                val = err_dict.get(comp, {}).get(axis, {}).get('var', np.nan)
                values.append(val if val is not None else np.nan)
            dim_values[f"{comp}_{axis}"] = np.array(values, dtype=float)

        # === 3. 创建画布 ===
        fig, ax_left = plt.subplots(figsize=(16, 8))
        ax_right = ax_left.twinx()

        title = error_labels.get(err_key, err_key)
        ax_left.set_title(f'{title}\n6维度误差方差波形 ',
                          fontsize=16, fontweight='bold', pad=30)

        lines_left, labels_left = [], []
        lines_right, labels_right = [], []

        for comp, axis, unit, color, use_right in dim_config:
            values = dim_values[f"{comp}_{axis}"]
            if len(values) == 0: continue

            ax = ax_right if use_right else ax_left
            target_list = lines_right if use_right else lines_left
            label_list = labels_right if use_right else labels_left

            line, = ax.plot(x_axis + 1, values, marker='o', markersize=5,
                            linewidth=1.5, color=color, label=f'{axis.upper()}{unit}',
                            markeredgewidth=0.8, markeredgecolor='white')
            target_list.append(line)
            label_list.append(f'{axis.upper()}{unit}')

            # 数值标注
            for i, v in enumerate(values):
                if not np.isnan(v) and i % 4 == 0 and v > 0.001:
                    offset = 0.0008 if use_right else 0.0015
                    ax.text(i + 1, v + offset, f'{v:.4f}', fontsize=8,
                            color=color, ha='center', va='bottom', fontweight='normal')

        # === 🎯 应用动态坐标轴范围 + 大字体配置 ===
        # 左轴（旋转方差）
        ax_left.set_xlabel('数据组别 (文件夹名称)', fontsize=13)  # 🔥 标签文字更新
        ax_left.set_ylabel(r'旋转误差方差 ($^\circ$$^2$)', fontsize=13, color='#1f77b4')
        ax_left.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)
        ax_left.set_xticks(x_axis + 1)
        # 🔑 核心改动: 使用解析后的文件夹名称 + 旋转30°更易读
        ax_left.set_xticklabels(x_labels, rotation=30, fontsize=11, ha='right')
        ax_left.set_ylim(rot_ylim)
        ax_left.grid(True, linestyle='--', alpha=0.4, axis='both')
        ax_left.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

        # 🎨 旋转稳定区间色带
        stable_rot_max = rot_ylim[1] * STABLE_BAND_RATIO
        if stable_rot_max > MIN_ROT_SPAN:
            ax_left.axhspan(0, stable_rot_max, color='green', alpha=0.05)

        # 右轴（平移方差）
        ax_right.set_ylabel(r'平移误差方差 ($mm^2$)', fontsize=13, color='#ff7f0e')
        ax_right.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=11)
        ax_right.set_ylim(trans_ylim)
        ax_right.grid(False)

        # 🎨 平移稳定区间色带
        stable_trans_max = trans_ylim[1] * STABLE_BAND_RATIO
        if stable_trans_max > MIN_TRANS_SPAN:
            ax_right.axhspan(0, stable_trans_max, color='green', alpha=0.05)

        # === 🔧 图例上置 + 大字体 ===
        lines_all = lines_left + lines_right
        labels_all = labels_left + labels_right
        if lines_all:
            ax_left.legend(lines_all, labels_all, loc='upper center',
                           bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=11,
                           framealpha=0.98, edgecolor='gray', fancybox=True)

        # === 右上角范围标注 + 大字体 ===
        range_info = f"rot: [{rot_ylim[1]:.3f}]  |  trans: [{trans_ylim[1]:.3f}]"
        ax_left.text(0.99, 0.87, range_info, transform=ax_left.transAxes, fontsize=9,
                     ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # # === 显式调整边距：横轴标签旋转30°需略增底部空间 ===
        # plt.subplots_adjust(left=0.09, right=0.92, top=0.88, bottom=0.12)
        #
        # out_path = os.path.join(save_dir, f"{err_key}_var_waveform_6dims.png")
        # plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        # plt.close(fig)
        print(f"  ✅ 已保存: {err_key}_var_waveform_6dims.png | rot_ylim={rot_ylim}, trans_ylim={trans_ylim}")

    print(f"🎨 绘图完成！共生成 {len(data_dict)} 张大字体高清方差波形图，保存至: {save_dir}")


def batch_process_data_plot(base_folder, save_dir):
    """
    批量处理主函数（保持原逻辑，仅调用新绘图函数）
    """
    os.makedirs(save_dir, exist_ok=True)

    # ================= 1. 配置常量 =================
    R_ctsz2osz_ct1 = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    R_ctsc2osc_ct2 = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    R_ctsz2osz = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 500], [0, 0, 0, 1]], dtype=np.float64)
    R_ctsc2osc = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 500], [0, 0, 0, 1]], dtype=np.float64)

    var_pa = [
        np.array([0.1533, 0.1132, 0.0952, 0.1311, 0.4423, 0.1775], dtype=np.float64),
        np.array([0.0567, 0.0370, 0.0459, 0.0842, 0.4745, 0.1262], dtype=np.float64),
        np.array([0.1197, 0.0702, 0.1016, 0.1364, 2.8470, 0.2270], dtype=np.float64)
    ]
    var_rlat = [
        np.array([0.1314, 0.1437, 0.0988, 0.5068, 0.1328, 0.2004], dtype=np.float64),
        np.array([0.0443, 0.0577, 0.0545, 0.4961, 0.1028, 0.1340], dtype=np.float64),
        np.array([0.0793, 0.1227, 0.1049, 3.2533, 0.1386, 0.2485], dtype=np.float64)
    ]
    var_r2p = [
        np.array([0.1309, 0.1450, 0.0993, 0.5593, 0.1857, 0.2570], dtype=np.float64),
        np.array([0.0437, 0.0582, 0.0551, 0.5303, 0.1311, 0.1608], dtype=np.float64),
        np.array([0.0782, 0.1237, 0.1061, 3.4051, 0.1769, 0.2905], dtype=np.float64)
    ]

    # ================= 2. 获取子文件夹 =================
    subfolders = sorted([d for d in os.listdir(base_folder)
                         if os.path.isdir(os.path.join(base_folder, d))])
    print(f"📁 在 {base_folder} 中找到 {len(subfolders)} 个子文件夹")

    error_keys = [
        'pa_mix1', 'rlat_mix1', 'rlat2pa_mix1', 'fusion_mix1',
        'pa_mix2', 'rlat_mix2', 'rlat2pa_mix2', 'fusion_mix2',
        'pa_mix3', 'rlat_mix3', 'rlat2pa_mix3', 'fusion_mix3'
    ]
    collected_errors = {k: [] for k in error_keys}

    # 🔵 [新增] 初始化融合结果存储列表（用于后续导出）
    fusion_mix1_res = []
    fusion_mix2_res = []
    fusion_mix3_res = []

    # ================= 3. 遍历处理 =================
    for idx, sub in enumerate(subfolders, 1):
        sub_path = os.path.join(base_folder, sub)
        csv_files = sorted([os.path.join(sub_path, f) for f in os.listdir(sub_path)
                            if f.lower().endswith('.csv')])

        if len(csv_files) != 6:
            print(f"⚠️ [{idx}/{len(subfolders)}] 跳过 {sub}: 预期6个CSV，实际 {len(csv_files)} 个")
            continue

        print(f"🔄 [{idx}/{len(subfolders)}] 处理: {sub}")
        try:
            data = extract_pose_named_data(csv_files)

            # 解包数据
            p1_pr = data['pa_mix1_pre_rota']
            p1_pt = data['pa_mix1_pre_trans']
            p1_tr = data['pa_mix1_tru_rota']
            p1_tt = data['pa_mix1_tru_trans']
            r1_pr = data['rlat_mix1_pre_rota']
            r1_pt = data['rlat_mix1_pre_trans']
            r1_tr = data['rlat_mix1_tru_rota']
            r1_tt = data['rlat_mix1_tru_trans']
            p2_pr = data['pa_mix2_pre_rota']
            p2_pt = data['pa_mix2_pre_trans']
            r2_pr = data['rlat_mix2_pre_rota']
            r2_pt = data['rlat_mix2_pre_trans']
            p3_pr = data['pa_mix3_pre_rota']
            p3_pt = data['pa_mix3_pre_trans']
            r3_pr = data['rlat_mix3_pre_rota']
            r3_pt = data['rlat_mix3_pre_trans']

            # 坐标变换
            p2_pr_a, p2_pt_a = get_base_noise(p2_pr, p2_pt, R_ctsz2osz_ct1)
            r2_pr_a, r2_pt_a = get_base_noise(r2_pr, r2_pt, R_ctsc2osc_ct2)
            p3_pr_a, p3_pt_a = get_base_noise(p3_pr, p3_pt, R_ctsz2osz_ct1)
            r3_pr_a, r3_pt_a = get_base_noise(r3_pr, r3_pt, R_ctsc2osc_ct2)

            # 光源变换
            R_sz_osc2osz = get_source_transform(
                pa_rota=p1_tr, pa_trans=p1_tt,
                rlat_rota=r1_tr, rlat_trans=r1_tt,
                R_ctsz2osz=R_ctsz2osz, R_ctsc2osc=R_ctsc2osc
            )

            # 统一至正位空间
            r1_pr_sz, r1_pt_sz = get_osz_noise(r1_pr, r1_pt, R_sz_osc2osz, R_ctsz2osz, R_ctsc2osc)
            r2_pr_sz, r2_pt_sz = get_osz_noise(r2_pr_a, r2_pt_a, R_sz_osc2osz, R_ctsz2osz, R_ctsc2osc)
            r3_pr_sz, r3_pt_sz = get_osz_noise(r3_pr_a, r3_pt_a, R_sz_osc2osz, R_ctsz2osz, R_ctsc2osc)

            # === 计算误差 ===
            # Mix1
            e_pa1 = compute_pose_errors_unknown_mean(p1_pr, p1_tr, p1_pt, p1_tt)
            e_rl1 = compute_pose_errors_unknown_mean(r1_pr, r1_tr, r1_pt, r1_tt)
            e_r2p1 = compute_pose_errors_unknown_mean(r1_pr_sz, p1_tr, r1_pt_sz, p1_tt)
            fu_pr1, fu_pt1 = reg_data_fusion(var_pa[0], var_r2p[0], p1_pr, p1_pt, r1_pr_sz, r1_pt_sz)
            e_fu1 = compute_pose_errors_unknown_mean(fu_pr1, p1_tr, fu_pt1, p1_tt)

            # Mix2
            e_pa2 = compute_pose_errors_unknown_mean(p2_pr_a, p1_tr, p2_pt_a, p1_tt)
            e_rl2 = compute_pose_errors_unknown_mean(r2_pr_a, r1_tr, r2_pt_a, r1_tt)
            e_r2p2 = compute_pose_errors_unknown_mean(r2_pr_sz, p1_tr, r2_pt_sz, p1_tt)
            fu_pr2, fu_pt2 = reg_data_fusion(var_pa[1], var_r2p[1], p2_pr_a, p2_pt_a, r2_pr_sz, r2_pt_sz)
            e_fu2 = compute_pose_errors_unknown_mean(fu_pr2, p1_tr, fu_pt2, p1_tt)

            # Mix3
            e_pa3 = compute_pose_errors_unknown_mean(p3_pr_a, p1_tr, p3_pt_a, p1_tt)
            e_rl3 = compute_pose_errors_unknown_mean(r3_pr_a, r1_tr, r3_pt_a, r1_tt)
            e_r2p3 = compute_pose_errors_unknown_mean(r3_pr_sz, p1_tr, r3_pt_sz, p1_tt)
            fu_pr3, fu_pt3 = reg_data_fusion(var_pa[2], var_r2p[2], p3_pr_a, p3_pt_a, r3_pr_sz, r3_pt_sz)
            e_fu3 = compute_pose_errors_unknown_mean(fu_pr3, p1_tr, fu_pt3, p1_tt)


            # 🔵 [新增] 收集本次计算的融合位姿结果
            fusion_mix1_res.append((sub, fu_pr1, fu_pt1))
            fusion_mix2_res.append((sub, fu_pr2, fu_pt2))
            fusion_mix3_res.append((sub, fu_pr3, fu_pt3))


            # 收集结果
            collected_errors['pa_mix1'].append(e_pa1)
            collected_errors['rlat_mix1'].append(e_rl1)
            collected_errors['rlat2pa_mix1'].append(e_r2p1)
            collected_errors['fusion_mix1'].append(e_fu1)
            collected_errors['pa_mix2'].append(e_pa2)
            collected_errors['rlat_mix2'].append(e_rl2)
            collected_errors['rlat2pa_mix2'].append(e_r2p2)
            collected_errors['fusion_mix2'].append(e_fu2)
            collected_errors['pa_mix3'].append(e_pa3)
            collected_errors['rlat_mix3'].append(e_rl3)
            collected_errors['rlat2pa_mix3'].append(e_r2p3)
            collected_errors['fusion_mix3'].append(e_fu3)


        except Exception as e:
            print(f"❌ [{idx}/{len(subfolders)}] {sub} 出错: {e}")
            continue

    # ================= 🔵 [新增] 独立保存融合结果到 CSV =================
    def _save_fusion_csv(records, filename, out_dir):
        if not records:
            print(f"⚠️ {filename} 无数据可保存")
            return
        filepath = os.path.join(out_dir, filename)

        # 自动推断维度生成表头
        first_pr, first_pt = records[0][1], records[0][2]
        pr_len = int(np.prod(first_pr.shape)) if hasattr(first_pr, 'shape') else len(first_pr)
        pt_len = int(np.prod(first_pt.shape)) if hasattr(first_pt, 'shape') else len(first_pt)

        header = ['subfolder'] + [f'fu_pr_{i}' for i in range(pr_len)] + [f'fu_pt_{i}' for i in range(pt_len)]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for sub, pr, pt in records:
                # 兼容 numpy array 和 list，统一展平后写入
                pr_flat = pr.flatten() if isinstance(pr, np.ndarray) else pr
                pt_flat = pt.flatten() if isinstance(pt, np.ndarray) else pt
                writer.writerow([sub] + list(pr_flat) + list(pt_flat))
        print(f"💾 已保存: {filepath}")

    _save_fusion_csv(fusion_mix1_res, 'fusion_mix1_results.csv', save_dir)
    _save_fusion_csv(fusion_mix2_res, 'fusion_mix2_results.csv', save_dir)
    _save_fusion_csv(fusion_mix3_res, 'fusion_mix3_results.csv', save_dir)
    # ====================================================================

    # # ================= 4. 绘图 =================
    # print("🎨 开始绘制6维度同图波形...")
    # plot_pose_error_waveforms(collected_errors, subfolders, save_dir)
    # print("🎉 全部完成！结果保存至:", save_dir)


# ================= 使用示例 =================
if __name__ == "__main__":
    BASE_DIR = "../data/uliver3_data_B"
    SAVE_DIR = "../data/Gauss_var_img/uliver6_mul_var_waveforms"
    batch_process_data_plot(BASE_DIR, SAVE_DIR)