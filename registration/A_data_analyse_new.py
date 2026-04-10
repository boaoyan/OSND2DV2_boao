import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from registration.math_process.data_fusion import get_source_transform, get_base_noise, get_osz_noise


def extract_pose_named_data(csv_paths,
                            rot_cols=['pre_rota_z', 'pre_rota_x', 'pre_rota_y'],
                            trans_cols=['pre_trans_x', 'pre_trans_y', 'pre_trans_z'],
                            tru_rot_cols=['tru_rota_z', 'tru_rota_x', 'tru_rota_y'],
                            tru_trans_cols=['tru_trans_x', 'tru_trans_y', 'tru_trans_z'],
                            strip_suffix='_model',  # 🆕 自动去除的后缀
                            lowercase=True):

    all_data = {}
    req_cols = rot_cols + trans_cols + tru_rot_cols + tru_trans_cols

    for csv_path in csv_paths:
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        if df.empty: continue

        pt_raw = df['pose_type'].iloc[0]
        mn_raw = df['model_name'].iloc[0]

        # 🆕 清理 model_name：去除指定后缀（如 _model, _v1 等）
        mn_clean = str(mn_raw).replace(strip_suffix, '') if strip_suffix else str(mn_raw)

        # 统一转小写 & 替换非法字符
        pt = re.sub(r'[^a-z0-9]', '_', pt_raw.lower().strip()).strip('_')
        mn = re.sub(r'[^a-z0-9]', '_', mn_clean.lower().strip()).strip('_')
        prefix = f"{pt}_{mn}"

        valid_df = df.dropna(subset=req_cols).copy()
        if valid_df.empty: continue

        all_data[f"{prefix}_pre_rota"] = valid_df[rot_cols].to_numpy(np.float64)
        all_data[f"{prefix}_pre_trans"] = valid_df[trans_cols].to_numpy(np.float64)
        all_data[f"{prefix}_tru_rota"] = valid_df[tru_rot_cols].to_numpy(np.float64)
        all_data[f"{prefix}_tru_trans"] = valid_df[tru_trans_cols].to_numpy(np.float64)

        print(f"✅ 提取: {pt_raw}/{mn_raw} -> {prefix}_pre_rota (shape: {all_data[f'{prefix}_pre_rota'].shape})")

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
                    angle_unit='rad'):  # 🆕 保留参数兼容旧调用，但内部已忽略
    """
    6-DoF 位姿加权融合（线性加权，旋转直接使用角度加权）

    Parameters
    ----------
    error1, error2 : dict 或 array
        两组预测的每自由度方差。若为 dict 需含 ['rot/trans'][x/y/z]['var']
    pre_rota1, pre_rota2 : array-like, shape (3,) or (N, 3)
        旋转预测值 [roll, pitch, yaw]（单位任意，直接线性平均）
    pre_trans1, pre_trans2 : array-like, shape (3,) or (N, 3)
        平移预测值 [tx, ty, tz]
    angle_unit : str
        保留参数以防旧接口报错，实际已忽略角度转换与圆周融合逻辑

    Returns
    -------
    fused_rot : np.ndarray, shape (3,) or (N, 3)
    fused_trans : np.ndarray, shape (3,) or (N, 3)
    """

    # 🔧 内部辅助：将 dict 或 array 统一转为 (6,) 方差数组
    def _parse_variance(err_input):
        if isinstance(err_input, dict):
            axes = ['x', 'y', 'z']
            try:
                r_vars = [err_input['rot'][ax]['var'] for ax in axes]
                t_vars = [err_input['trans'][ax]['var'] for ax in axes]
                return np.array(r_vars + t_vars)
            except KeyError as e:
                raise ValueError(f"误差字典结构缺失键: {e}。需包含 rot/trans 下的 x,y,z.var")
        else:
            return np.asarray(err_input, dtype=float)

    # 统一转为 2D 方便向量化计算 (N, D)
    var1 = np.atleast_2d(_parse_variance(error1))
    var2 = np.atleast_2d(_parse_variance(error2))
    r1   = np.atleast_2d(np.asarray(pre_rota1, dtype=float))
    t1   = np.atleast_2d(np.asarray(pre_trans1, dtype=float))
    r2   = np.atleast_2d(np.asarray(pre_rota2, dtype=float))
    t2   = np.atleast_2d(np.asarray(pre_trans2, dtype=float))

    # 维度校验
    if var1.shape != var2.shape or var1.shape[1] != 6:
        raise ValueError("Variance must be shape (6,) or (N, 6)")
    if r1.shape != r2.shape or t1.shape != t2.shape:
        raise ValueError("Rotation/Translation shapes must match")
    if r1.shape[1] != 3 or t1.shape[1] != 3:
        raise ValueError("Rot/Trans must be shape (3,) or (N, 3)")

    # 🆕 已忽略角度单位转换，直接按原始数值计算

    # 1. 计算权重 (向量化)
    total_var = var1 + var2
    total_var = np.where(total_var == 0, 1e-12, total_var)  # 防除零
    w1 = var2 / total_var  # 对应 error1 的权重
    w2 = var1 / total_var  # 对应 error2 的权重 (w1 + w2 = 1)

    # 拆分旋转/平移权重
    w1_rot, w2_rot = w1[:, :3], w2[:, :3]
    w1_trans, w2_trans = w1[:, 3:], w2[:, 3:]

    # 2. 旋转融合：🆕 直接线性加权（忽略圆周特性）
    fused_rot = w1_rot * r1 + w2_rot * r2

    # 3. 平移融合：线性加权
    fused_trans = w1_trans * t1 + w2_trans * t2

    # 若输入为单样本，恢复 1D 形状
    if fused_rot.shape[0] == 1:
        return fused_rot.squeeze(), fused_trans.squeeze()
    return fused_rot, fused_trans


def plot_error_bars(results, model_name,
                    rot_units='deg', trans_units='mm',
                    save_path=None, figsize=(9, 4),
                    block=False,
                    value_fmt='{:.4f}',        # 🆕 方差显示格式（单位²）
                    show_std_in_label=False,   # 🆕 是否在标签中附加 √var
                    label_fontsize=8):
    """
    绘制方差直方图（旋转 + 平移并排），柱高 = variance，柱顶显示数值

    Parameters
    ----------
    results : dict
        结构：{'rot': {'x': {'mean', 'var'}, ...}, 'trans': {...}}
    model_name : str
        模型/方法名称，用于标题和窗口标识
    rot_units / trans_units : str
        原始物理单位（如 'deg', 'mm'），方差单位自动为 {unit}²
    value_fmt : str
        方差数值格式，如 '{:.4f}' → "0.1459"
    show_std_in_label : bool
        若为 True，标签显示 "var=0.146 (σ=0.382)"；否则仅显示 var
    label_fontsize : int
        柱顶标签字体大小
    """
    # 为每个模型创建独立窗口，避免复用
    fig, (ax_rot, ax_trans) = plt.subplots(1, 2, figsize=figsize, num=model_name)
    fig.suptitle(f'Error Variance - {model_name}', fontsize=13, fontweight='bold', y=0.98)

    colors = {'x': '#2E86AB', 'y': '#A23B72', 'z': '#F18F01'}
    axes_order = ['x', 'y', 'z']

    # === 旋转方差 ===
    rot_vars = [results['rot'][ax]['var'] for ax in axes_order if ax in results['rot'] and 'var' in results['rot'][ax]]
    rot_axes = [ax for ax in axes_order if ax in results['rot'] and 'var' in results['rot'][ax]]
    x_rot = np.arange(len(rot_vars))

    if rot_vars:
        bars_rot = ax_rot.bar(x_rot, rot_vars,
                              color=[colors[ax] for ax in rot_axes],
                              edgecolor='black', linewidth=0.6)
        ax_rot.set_xticks(x_rot)
        ax_rot.set_xticklabels([ax.upper() for ax in rot_axes], fontsize=10)

        # 🆕 柱顶标签：var 或 "var (σ=std)"
        for bar, var_val in zip(bars_rot, rot_vars):
            height = bar.get_height()
            std_val = np.sqrt(var_val)
            if show_std_in_label:
                label_text = f'{value_fmt.format(var_val)} (σ={value_fmt.format(std_val)})'
            else:
                label_text = value_fmt.format(var_val)
            # 标签位置：柱顶上方 2% 最大方差高度
            offset = 0.02 * max(rot_vars) if max(rot_vars) > 0 else 0.01
            ax_rot.text(bar.get_x() + bar.get_width() / 2, height + offset,
                        label_text,
                        ha='center', va='bottom', fontsize=label_fontsize, fontweight='normal')

    ax_rot.set_ylabel(f'Variance ({rot_units}²)', fontsize=9)  # 🆕 单位平方
    ax_rot.set_title('Rotation', fontsize=10, fontweight='bold')
    ax_rot.grid(axis='y', linestyle='--', alpha=0.4)
    ax_rot.set_axisbelow(True)
    # 🆕 y 轴上限：预留 25% 空间给标签
    ax_rot.set_ylim(0, max(rot_vars) * 1.25 if rot_vars else 1)

    # === 平移方差 ===
    trans_vars = [results['trans'][ax]['var'] for ax in axes_order if ax in results['trans'] and 'var' in results['trans'][ax]]
    trans_axes = [ax for ax in axes_order if ax in results['trans'] and 'var' in results['trans'][ax]]
    x_trans = np.arange(len(trans_vars))

    if trans_vars:
        bars_trans = ax_trans.bar(x_trans, trans_vars,
                                  color=[colors[ax] for ax in trans_axes],
                                  edgecolor='black', linewidth=0.6)
        ax_trans.set_xticks(x_trans)
        ax_trans.set_xticklabels([ax.upper() for ax in trans_axes], fontsize=10)

        for bar, var_val in zip(bars_trans, trans_vars):
            height = bar.get_height()
            std_val = np.sqrt(var_val)
            if show_std_in_label:
                label_text = f'{value_fmt.format(var_val)} (σ={value_fmt.format(std_val)})'
            else:
                label_text = value_fmt.format(var_val)
            offset = 0.02 * max(trans_vars) if max(trans_vars) > 0 else 0.01
            ax_trans.text(bar.get_x() + bar.get_width() / 2, height + offset,
                          label_text,
                          ha='center', va='bottom', fontsize=label_fontsize, fontweight='normal')

    ax_trans.set_ylabel(f'Variance ({trans_units}²)', fontsize=9)  # 🆕 单位平方
    ax_trans.set_title('Translation', fontsize=10, fontweight='bold')
    ax_trans.grid(axis='y', linestyle='--', alpha=0.4)
    ax_trans.set_axisbelow(True)
    ax_trans.set_ylim(0, max(trans_vars) * 1.25 if trans_vars else 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show(block=block)
    if not block:
        plt.pause(0.001)

    return fig



if __name__ == '__main__':

    # === 1. 配置 4 个 CSV 路径 ===
    csv_files = [
        '../data/mix2_model_output/model_output_pa_mix1.csv',
        '../data/mix2_model_output/model_output_rlat_mix1.csv',
        '../data/mix2_model_output/model_output_pa_mix2.csv',
        '../data/mix2_model_output/model_output_rlat_mix2.csv'
    ]
    R_ctsz2osz_ct1 = np.array([[-1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]],dtype=np.float64)
    R_ctsc2osc_ct2 = np.array([[0, -1, 0, 0],
                                 [0, 0, 1, 0],
                                 [-1, 0, 0, 0],
                                 [0, 0, 0, 1]], dtype=np.float64)

    R_ctsz2osz = np.array([[-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 500],
                               [0, 0, 0, 1]],dtype=np.float64)
    R_ctsc2osc = np.array([[0, -1, 0, 0],
                               [0, 0, 1, 0],
                               [-1, 0, 0, 500],
                               [0, 0, 0, 1]], dtype=np.float64)

    print("🔄 批量计算姿态误差中...")
    data = extract_pose_named_data(csv_files)

    # 数据提取
    pa_mix1_pre_rota = data['pa_mix1_pre_rota']
    pa_mix1_pre_trans = data['pa_mix1_pre_trans']
    pa_mix1_tru_rota = data['pa_mix1_tru_rota']
    pa_mix1_tru_trans = data['pa_mix1_tru_trans']

    rlat_mix1_pre_rota = data['rlat_mix1_pre_rota']
    rlat_mix1_pre_trans = data['rlat_mix1_pre_trans']
    rlat_mix1_tru_rota = data['rlat_mix1_tru_rota']
    rlat_mix1_tru_trans = data['rlat_mix1_tru_trans']

    pa_mix2_pre_rota = data['pa_mix2_pre_rota']
    pa_mix2_pre_trans = data['pa_mix2_pre_trans']
    pa_mix2_tru_rota = data['pa_mix2_tru_rota']
    pa_mix2_tru_trans = data['pa_mix2_tru_trans']

    rlat_mix2_pre_rota = data['rlat_mix2_pre_rota']
    rlat_mix2_pre_trans = data['rlat_mix2_pre_trans']
    rlat_mix2_tru_rota = data['rlat_mix2_tru_rota']
    rlat_mix2_tru_trans = data['rlat_mix2_tru_trans']

    # 统一噪声和标签到各视角实际ct坐标系下
    pa_mix2_pre_rota_a, pa_mix2_pre_trans_a = get_base_noise(
                                                pa_mix2_pre_rota,
                                                pa_mix2_pre_trans,
                                                R_ctsz2osz_ct1)
    # pa_mix2_tru_rota_a, pa_mix2_tru_trans_a = get_base_noise(
    #                                             pa_mix2_tru_rota,
    #                                             pa_mix2_tru_trans,
    #                                             R_ctsz2osz_ct1)
    rlat_mix2_pre_rota_a, rlat_mix2_pre_trans_a = get_base_noise(
                                                    rlat_mix2_pre_rota,
                                                    rlat_mix2_pre_trans,
                                                    R_ctsc2osc_ct2)
    # rlat_mix2_tru_rota_a, rlat_mix2_tru_trans_a = get_base_noise(
    #                                                 rlat_mix2_tru_rota,
    #                                                 rlat_mix2_tru_trans,
    #                                                 R_ctsc2osc_ct2)

    # 获取加入噪声后正位光源与侧位光源之间的变换关系
    R_sz_osc2osz = get_source_transform(
                    pa_rota=pa_mix1_tru_rota,
                    pa_trans=pa_mix1_tru_trans,
                    rlat_rota=rlat_mix1_tru_rota,
                    rlat_trans=rlat_mix1_tru_trans,
                    R_ctsz2osz=R_ctsz2osz,
                    R_ctsc2osc=R_ctsc2osc) # (N,4,4)

    # 统一所有噪声到正位视角的空间姿态下
    rlat_mix1_pre_rota_sz, rlat_mix1_pre_trans_sz = get_osz_noise(
                                                        rlat_mix1_pre_rota,
                                                        rlat_mix1_pre_trans,
                                                        R_sz_osc2osz,
                                                        R_ctsz2osz,
                                                        R_ctsc2osc)
    # rlat_mix1_tru_rota_sz, rlat_mix1_tru_trans_sz = get_osz_noise(
    #                                                     rlat_mix1_tru_rota,
    #                                                     rlat_mix1_tru_trans,
    #                                                     R_sz_osc2osz,
    #                                                     R_ctsz2osz,
    #                                                     R_ctsc2osc)

    rlat_mix2_pre_rota_sz, rlat_mix2_pre_trans_sz = get_osz_noise(
                                                        rlat_mix2_pre_rota_a,
                                                        rlat_mix2_pre_trans_a,
                                                        R_sz_osc2osz,
                                                        R_ctsz2osz,
                                                        R_ctsc2osc)
    # rlat_mix2_tru_rota_sz, rlat_mix2_tru_trans_sz = get_osz_noise(
    #                                                     rlat_mix2_tru_rota_a,
    #                                                     rlat_mix2_tru_trans_a,
    #                                                     R_sz_osc2osz,
    #                                                     R_ctsz2osz,
    #                                                     R_ctsc2osc)

    # # 计算mix1模型姿态误差
    error_pa_mix1 = compute_pose_errors_unknown_mean(pre_rota=pa_mix1_pre_rota,
                                               tru_rota=pa_mix1_tru_rota,
                                               pre_trans=pa_mix1_pre_trans,
                                               tru_trans=pa_mix1_tru_trans)
    error_rlat_mix1 = compute_pose_errors_unknown_mean(pre_rota=rlat_mix1_pre_rota,
                                               tru_rota=rlat_mix1_tru_rota,
                                               pre_trans=rlat_mix1_pre_trans,
                                               tru_trans=rlat_mix1_tru_trans)
    error_rlat2pa_mix1 = compute_pose_errors_unknown_mean(pre_rota=rlat_mix1_pre_rota_sz,
                                                 tru_rota=pa_mix1_tru_rota,
                                                 pre_trans=rlat_mix1_pre_trans_sz,
                                                 tru_trans=pa_mix1_tru_trans)
    # # 计算mix1融合后的姿态误差
    fusion_pre_rota_mix1, fusion_pre_trans_mix1 = reg_data_fusion(error1=error_pa_mix1,
                                                        error2=error_rlat2pa_mix1,
                                                        pre_rota1=pa_mix1_pre_rota,
                                                        pre_trans1=pa_mix1_pre_trans,
                                                        pre_rota2=rlat_mix1_pre_rota_sz,
                                                        pre_trans2=rlat_mix1_pre_trans_sz)

    error_fusion_mix1 = compute_pose_errors_unknown_mean(pre_rota=fusion_pre_rota_mix1,
                                                         tru_rota=pa_mix1_tru_rota,
                                                         pre_trans=fusion_pre_trans_mix1,
                                                         tru_trans=pa_mix1_tru_trans)

    # # 计算mix2模型姿态误差
    error_pa_mix2 = compute_pose_errors_unknown_mean(pre_rota=pa_mix2_pre_rota_a,
                                               tru_rota=pa_mix1_tru_rota,
                                               pre_trans=pa_mix2_pre_trans_a,
                                               tru_trans=pa_mix1_tru_trans)
    error_rlat_mix2 = compute_pose_errors_unknown_mean(pre_rota=rlat_mix2_pre_rota_a,
                                                 tru_rota=rlat_mix1_tru_rota,
                                                 pre_trans=rlat_mix2_pre_trans_a,
                                                 tru_trans=rlat_mix1_tru_trans)
    error_rlat2pa_mix2 = compute_pose_errors_unknown_mean(pre_rota=rlat_mix2_pre_rota_sz,
                                                    tru_rota=pa_mix1_tru_rota,
                                                    pre_trans=rlat_mix2_pre_trans_sz,
                                                    tru_trans=pa_mix1_tru_trans)
    # # 计算mix2融合后的姿态误差
    fusion_pre_rota_mix2, fusion_pre_trans_mix2 = reg_data_fusion(error1=error_pa_mix2,
                                                        error2=error_rlat2pa_mix2,
                                                        pre_rota1=pa_mix2_pre_rota_a,
                                                        pre_trans1=pa_mix2_pre_trans_a,
                                                        pre_rota2=rlat_mix2_pre_rota_sz,
                                                        pre_trans2=rlat_mix2_pre_trans_sz)

    error_fusion_mix2 = compute_pose_errors_unknown_mean(pre_rota=fusion_pre_rota_mix2,
                                                         tru_rota=pa_mix1_tru_rota,
                                                         pre_trans=fusion_pre_trans_mix2,
                                                         tru_trans=pa_mix1_tru_trans)
    # 绘图
    # ✅ 方式1：直接循环调用（推荐）
    for name, err in [
        ('pa_mix1', error_pa_mix1),
        ('rlat_mix1', error_rlat_mix1),
        ('rlat2pa_mix1', error_rlat2pa_mix1),
        ('fusion_mix1', error_fusion_mix1),
        ('pa_mix2', error_pa_mix2),
        ('rlat_mix2', error_rlat_mix2),
        ('rlat2pa_mix2', error_rlat2pa_mix2),
        ('fusion_mix2', error_fusion_mix2)
    ]:
        plot_error_bars(err, name, block=False)  # 🆕 非阻塞模式

    # 🆕 关键：最后添加一个阻塞式 show() 保持所有窗口显示
    # （否则主程序退出后窗口会被关闭）
    plt.show()


    print("✅ 批量计算姿态误差完成！")

