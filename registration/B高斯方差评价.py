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


def save_pose_arrays_to_csv(csv_path, **pose_dict):
    """
    将多组 (N,3) 旋转/平移数组保存为 CSV
    参数:
        csv_path: 保存路径 (str)
        **pose_dict: 关键字参数，格式为 prefix=(rota_array, trans_array)
                     例如: pa_mix1=(pa_mix1_tru_rota, pa_mix1_tru_trans)
    返回:
        pd.DataFrame: 生成的数据表（便于后续查看）
    """
    df_data = {}
    target_n = None

    for prefix, (rota, trans) in pose_dict.items():
        rota = np.asarray(rota)
        trans = np.asarray(trans)

        # 🔍 维度校验
        if not (rota.ndim == 2 and trans.ndim == 2 and rota.shape[1] == trans.shape[1] == 3):
            raise ValueError(f"❌ {prefix}: 数组必须为 (N, 3) 形状，当前为 rota{rota.shape} / trans{trans.shape}")
        if target_n is None:
            target_n = rota.shape[0]
        elif rota.shape[0] != target_n:
            raise ValueError(f"❌ {prefix}: 样本数({rota.shape[0]})与首组({target_n})不一致，无法对齐保存")

        # 📦 按约定顺序展开列 (旋转: Z,X,Y | 平移: X,Y,Z)
        df_data[f'{prefix}_rota_z'] = rota[:, 0]
        df_data[f'{prefix}_rota_x'] = rota[:, 1]
        df_data[f'{prefix}_rota_y'] = rota[:, 2]
        df_data[f'{prefix}_trans_x'] = trans[:, 0]
        df_data[f'{prefix}_trans_y'] = trans[:, 1]
        df_data[f'{prefix}_trans_z'] = trans[:, 2]

    # 💾 导出 CSV (保留6位小数，避免科学计数法)
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False, float_format='%.6f')

    print(f"✅ 已成功保存 {len(pose_dict)} 组数据至: {csv_path}")
    print(f"📊 DataFrame 形状: {df.shape} | 列顺序: {list(df.columns)}")
    return df

if __name__ == '__main__':

    # === 1. 配置 4 个 CSV 路径 ===
    # csv_files = [
    #     '../data/all_model_data/8000_pa_mix1_model.csv',
    #     '../data/all_model_data/8000_rlat_mix1_model.csv',
    #     '../data/all_model_data/8000_pa_mix2_model.csv',
    #     '../data/all_model_data/8000_rlat_mix2_model.csv',
    #     '../data/all_model_data/8000_pa_mix3_model.csv',
    #     '../data/all_model_data/8000_rlat_mix3_model.csv'
    # ]
    # csv_files = [
    #     '../data/uliver6_mul_data/8000_1/8000_copy1_pa_mix1_model.csv',
    #     '../data/uliver6_mul_data/8000_1/8000_copy1_rlat_mix1_model.csv',
    #     '../data/uliver6_mul_data/8000_1/8000_copy1_pa_mix2_model.csv',
    #     '../data/uliver6_mul_data/8000_1/8000_copy1_rlat_mix2_model.csv',
    #     '../data/uliver6_mul_data/8000_1/8000_copy1_pa_mix3_model.csv',
    #     '../data/uliver6_mul_data/8000_1/8000_copy1_rlat_mix3_model.csv'
    # ]
    csv_files = ['../data/uliver6_2dof_data/8000_copy1_pa_mix1_model.csv',
                '../data/uliver6_2dof_data/8000_copy1_rlat_mix1_model.csv',
                '../data/uliver6_2dof_data/8000_copy1_pa_mix2_model.csv',
                 '../data/uliver6_2dof_data/8000_copy1_rlat_mix2_model.csv'
                 # '../data/uliver6_2dof_data/8000_copy1_pa_mix3_model.csv',
                 # '../data/uliver6_2dof_data/8000_copy1_rlat_mix3_model.csv'
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
    # # spine107_img.nii.gz先验方差
    # var_pa_mix1 = np.array([0.1431, 0.1256, 0.1480, 0.1259, 0.3615, 0.0916], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    # var_rlat_mix1 = np.array([0.1259, 0.1529, 0.1408, 0.4875, 0.1050, 0.0779], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    # var_r2p_mix1 = np.array([0.1256, 0.1547, 0.1410, 0.5684, 0.1552, 0.1260], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    #
    # var_pa_mix2 = np.array([0.0588, 0.0277, 0.0406, 0.0442, 0.4377, 0.0530], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    # var_rlat_mix2 = np.array([0.0288, 0.0579, 0.0424, 0.04176, 0.0472, 0.0479], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_r2p_mix2 = np.array([0.0286, 0.0585, 0.0426, 0.4493, 0.0616, 0.0668], dtype=np.float64)
    #
    # var_pa_mix3 = np.array([0.1185, 0.0522, 0.1232, 0.0696, 0.5706, 0.0908], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_rlat_mix3 = np.array([0.550, 0.1206, 0.1210, 0.5540, 0.0805, 0.0812], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_r2p_mix3 = np.array([0.0553, 0.1223, 0.1207, 0.6104, 0.1118, 0.1131], dtype=np.float64)

    # uniformed_liver_3.nii.gz先验方差
    var_pa_mix1 = np.array([0.1533, 0.1132, 0.0952, 0.1311, 0.4423, 0.1775], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    var_rlat_mix1 = np.array([0.1314, 0.1437, 0.0988, 0.5068, 0.1328, 0.2004], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    var_r2p_mix1 = np.array([0.1309, 0.1450, 0.0993, 0.5593, 0.1857, 0.2570], dtype=np.float64) # rx,ry,rz,tx,ty,tz

    var_pa_mix2 = np.array([0.0567, 0.0370, 0.0459, 0.0842, 0.4745, 0.1262], dtype=np.float64) # rx,ry,rz,tx,ty,tz
    var_rlat_mix2 = np.array([0.0443, 0.0577, 0.0545, 0.4961, 0.1028, 0.1340], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    var_r2p_mix2 = np.array([0.0437,	0.0582,	0.0551,	0.5303,	0.1311,	0.1608], dtype=np.float64)

    var_pa_mix3 = np.array([0.1197, 0.0702, 0.1016, 0.1364, 2.8470, 0.2270], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    var_rlat_mix3 = np.array([0.0793, 0.1227, 0.1049, 3.2533, 0.1386, 0.2485], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    var_r2p_mix3 = np.array([0.0782,	0.1237,	0.1061,	3.4051,	0.1769,	0.2905], dtype=np.float64)

    # # # uniformed_liver_6.nii.gz先验方差
    # var_pa_mix1 = np.array([0.1420, 0.1144, 0.0999, 0.1392, 0.4655, 0.1783], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_rlat_mix1 = np.array([0.1340, 0.1628, 0.0963, 1.1248, 0.1386, 0.1885], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_r2p_mix1 = np.array([0.1231, 0.1641, 0.0972, 1.2057, 0.1929, 0.2332], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    #
    # var_pa_mix2 = np.array([0.0805, 0.0374, 0.0540, 0.0988, 2.1465, 0.2136], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_rlat_mix2 = np.array([0.0418, 0.0861, 0.0641, 1.6101, 0.1118, 0.2306], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_r2p_mix2 = np.array([0.0413, 0.0866, 0.0645, 1.6884, 0.1448, 0.2628], dtype=np.float64)
    # mix3_model
    # var_pa_mix3 = np.array([0.0739, 0.0407, 0.0567, 0.1169, 2.5242, 0.1750], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_rlat_mix3 = np.array([0.0490, 0.0953, 0.0585, 2.5771, 0.1156, 0.1611], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_r2p_mix3 = np.array([0.0487, 0.0957, 0.0589, 2.6776, 0.1473, 0.1993], dtype=np.float64)
    # # mix3_diff_model
    # var_pa_mix3 = np.array([0.0679, 0.0341, 0.0549, 0.0828, 2.5020, 0.1881], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_rlat_mix3 = np.array([0.0415, 0.0916, 0.0525, 2.7259, 0.0919, 0.1977], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    # var_r2p_mix3 = np.array([0.0411, 0.0919, 0.0529, 2.7924, 0.1163, 0.2350], dtype=np.float64)
    # mix3_mul_model
    var_pa_mix3 = np.array([4.7047, 0.05856, 1.6702, 0.9584, 6.4125, 2.8086], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    var_rlat_mix3 = np.array([0.4316, 4.5431, 2.3956, 8.7367, 1.4241, 3.0262], dtype=np.float64)  # rx,ry,rz,tx,ty,tz
    var_r2p_mix3 = np.array([0.4130, 4.5192, 2.4020, 7.2184, 0.9489, 2.3144], dtype=np.float64)


    print("🔄 批量计算姿态误差中...")
    data = extract_pose_named_data(csv_files)
    print("data 的类型:", type(data))
    if isinstance(data, dict):
        print("实际存在的键:", list(data.keys()))
    elif hasattr(data, "columns"):  # 如果是 pandas DataFrame
        print("实际存在的列名:", list(data.columns))
    else:
        print("data 结构:", data)
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
    # # #
    # pa_mix3_pre_rota = data['pa_mix3_pre_rota']
    # pa_mix3_pre_trans = data['pa_mix3_pre_trans']
    # pa_mix3_tru_rota = data['pa_mix3_tru_rota']
    # pa_mix3_tru_trans = data['pa_mix3_tru_trans']
    # 
    # rlat_mix3_pre_rota = data['rlat_mix3_pre_rota']
    # rlat_mix3_pre_trans = data['rlat_mix3_pre_trans']
    # rlat_mix3_tru_rota = data['rlat_mix3_tru_rota']
    # rlat_mix3_tru_trans = data['rlat_mix3_tru_trans']

    # # 统一噪声和标签到各视角实际ct坐标系下
    pa_mix2_pre_rota_a, pa_mix2_pre_trans_a = get_base_noise(
                                                pa_mix2_pre_rota,
                                                pa_mix2_pre_trans,
                                                R_ctsz2osz_ct1)
    pa_mix2_tru_rota_a, pa_mix2_tru_trans_a = get_base_noise(
                                                pa_mix2_tru_rota,
                                                pa_mix2_tru_trans,
                                                R_ctsz2osz_ct1)
    rlat_mix2_pre_rota_a, rlat_mix2_pre_trans_a = get_base_noise(
                                                rlat_mix2_pre_rota,
                                                rlat_mix2_pre_trans,
                                                R_ctsc2osc_ct2)
    rlat_mix2_tru_rota_a, rlat_mix2_tru_trans_a = get_base_noise(
                                                rlat_mix2_tru_rota,
                                                rlat_mix2_tru_trans,
                                                R_ctsc2osc_ct2)
    #
    # #
    # pa_mix3_pre_rota_a, pa_mix3_pre_trans_a = get_base_noise(
    #                                             pa_mix3_pre_rota,
    #                                             pa_mix3_pre_trans,
    #                                             R_ctsz2osz_ct1)
    # pa_mix3_tru_rota_a, pa_mix3_tru_trans_a = get_base_noise(
    #                                             pa_mix3_tru_rota,
    #                                             pa_mix3_tru_trans,
    #                                             R_ctsz2osz_ct1)
    # rlat_mix3_pre_rota_a, rlat_mix3_pre_trans_a = get_base_noise(
    #                                             rlat_mix3_pre_rota,
    #                                             rlat_mix3_pre_trans,
    #                                             R_ctsc2osc_ct2)
    # rlat_mix3_tru_rota_a, rlat_mix3_tru_trans_a = get_base_noise(
    #                                             rlat_mix3_tru_rota,
    #                                             rlat_mix3_tru_trans,
    #                                             R_ctsc2osc_ct2)

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
    rlat_mix1_tru_rota_sz, rlat_mix1_tru_trans_sz = get_osz_noise(
                                                        rlat_mix1_tru_rota,
                                                        rlat_mix1_tru_trans,
                                                        R_sz_osc2osz,
                                                        R_ctsz2osz,
                                                        R_ctsc2osc)

    rlat_mix2_pre_rota_sz, rlat_mix2_pre_trans_sz = get_osz_noise(
                                                        rlat_mix2_pre_rota_a,
                                                        rlat_mix2_pre_trans_a,
                                                        R_sz_osc2osz,
                                                        R_ctsz2osz,
                                                        R_ctsc2osc)
    rlat_mix2_tru_rota_sz, rlat_mix2_tru_trans_sz = get_osz_noise(
                                                        rlat_mix2_tru_rota_a,
                                                        rlat_mix2_tru_trans_a,
                                                        R_sz_osc2osz,
                                                        R_ctsz2osz,
                                                        R_ctsc2osc)
    # #
    # rlat_mix3_pre_rota_sz, rlat_mix3_pre_trans_sz = get_osz_noise(
    #                                                     rlat_mix3_pre_rota_a,
    #                                                     rlat_mix3_pre_trans_a,
    #                                                     R_sz_osc2osz,
    #                                                     R_ctsz2osz,
    #                                                     R_ctsc2osc)
    # rlat_mix3_tru_rota_sz, rlat_mix3_tru_trans_sz = get_osz_noise(
    #                                                     rlat_mix3_tru_rota_a,
    #                                                     rlat_mix3_tru_trans_a,
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
    # # # # # # # 计算mix1融合后的姿态误差
    # fusion_pre_rota_mix1, fusion_pre_trans_mix1 = reg_data_fusion(error1=var_pa_mix1,
    #                                                     error2=var_r2p_mix1,
    #                                                     pre_rota1=pa_mix1_pre_rota,
    #                                                     pre_trans1=pa_mix1_pre_trans,
    #                                                     pre_rota2=rlat_mix1_pre_rota_sz,
    #                                                     pre_trans2=rlat_mix1_pre_trans_sz)
    # 
    # error_fusion_mix1 = compute_pose_errors_unknown_mean(pre_rota=fusion_pre_rota_mix1,
    #                                                      tru_rota=pa_mix1_tru_rota,
    #                                                      pre_trans=fusion_pre_trans_mix1,
    #                                                      tru_trans=pa_mix1_tru_trans)

    # # # # 计算mix2模型姿态误差
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
    # # # # # # # # 计算mix2融合后的姿态误差
    # fusion_pre_rota_mix2, fusion_pre_trans_mix2 = reg_data_fusion(error1=var_pa_mix2,
    #                                                     error2=var_r2p_mix2,
    #                                                     pre_rota1=pa_mix2_pre_rota_a,
    #                                                     pre_trans1=pa_mix2_pre_trans_a,
    #                                                     pre_rota2=rlat_mix2_pre_rota_sz,
    #                                                     pre_trans2=rlat_mix2_pre_trans_sz)
    # 
    # error_fusion_mix2 = compute_pose_errors_unknown_mean(pre_rota=fusion_pre_rota_mix2,
    #                                                      tru_rota=pa_mix1_tru_rota,
    #                                                      pre_trans=fusion_pre_trans_mix2,
    #                                                      tru_trans=pa_mix1_tru_trans)
    #
    # # # # # # # 计算mix3模型姿态误差
    # error_pa_mix3 = compute_pose_errors_unknown_mean(pre_rota=pa_mix3_pre_rota_a,
    #                                                  tru_rota=pa_mix1_tru_rota,
    #                                                  pre_trans=pa_mix3_pre_trans_a,
    #                                                  tru_trans=pa_mix1_tru_trans)
    # error_rlat_mix3 = compute_pose_errors_unknown_mean(pre_rota=rlat_mix3_pre_rota_a,
    #                                              tru_rota=rlat_mix1_tru_rota,
    #                                              pre_trans=rlat_mix3_pre_trans_a,
    #                                              tru_trans=rlat_mix1_tru_trans)
    # error_rlat2pa_mix3 = compute_pose_errors_unknown_mean(pre_rota=rlat_mix3_pre_rota_sz,
    #                                                 tru_rota=pa_mix1_tru_rota,
    #                                                 pre_trans=rlat_mix3_pre_trans_sz,
    #                                                 tru_trans=pa_mix1_tru_trans)
    # # # # # # # # 计算mix3融合后的姿态误差
    # fusion_pre_rota_mix3, fusion_pre_trans_mix3 = reg_data_fusion(error1=var_pa_mix3,
    #                                                     error2=var_r2p_mix3,
    #                                                     pre_rota1=pa_mix3_pre_rota_a,
    #                                                     pre_trans1=pa_mix3_pre_trans_a,
    #                                                     pre_rota2=rlat_mix3_pre_rota_sz,
    #                                                     pre_trans2=rlat_mix3_pre_trans_sz)
    # 
    # error_fusion_mix3 = compute_pose_errors_unknown_mean(pre_rota=fusion_pre_rota_mix3,
    #                                                      tru_rota=pa_mix1_tru_rota,
    #                                                      pre_trans=fusion_pre_trans_mix3,
    #                                                      tru_trans=pa_mix1_tru_trans)
    # #
    # # # # 绘图
    # # # ✅ 方式1：直接循环调用（推荐）
    output_dir = "../data/data_img/uliver6_2dof_img/8000_1"
    os.makedirs(output_dir, exist_ok=True)

    for name, err in [
        ('pa_mix1', error_pa_mix1),
        ('rlat_mix1', error_rlat_mix1),
        ('rlat2pa_mix1', error_rlat2pa_mix1),
        # ('fusion_mix1', error_fusion_mix1),
        ('pa_mix2', error_pa_mix2),
        ('rlat_mix2', error_rlat_mix2),
        ('rlat2pa_mix2', error_rlat2pa_mix2),
        # ('fusion_mix2', error_fusion_mix2),
        # ('pa_mix3', error_pa_mix3),
        # ('rlat_mix3', error_rlat_mix3),
        # ('rlat2pa_mix3', error_rlat2pa_mix3),
        # ('fusion_mix3', error_fusion_mix3)
    ]:
        save_path = os.path.join(output_dir, f"{name}.png")

        # plot_error_bars(err, name, block=False)
        plot_error_bars(err, name, block=False, save_path=save_path)  # 🆕 非阻塞模式

    # 🆕 关键：最后添加一个阻塞式 show() 保持所有窗口显示
    # （否则主程序退出后窗口会被关闭）
    plt.show()


    print("✅ 批量计算姿态误差完成！")