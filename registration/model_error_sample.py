import csv
from datetime import datetime

import os
import pandas as pd
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from registration.grid_efficientnet.grid_efficientb0_model import GridModel

from registration.label_transform_mix import LabelTransformMix
from registration.projector.drr import DRR
from registration.projector.read_data import read
from registration.projector.pose import convert

def norm_img(img):
    # img: [B, 1, H, W]
    B, C, H, W = img.shape
    img_flat = img.view(B, -1)  # [B, H*W]

    img_min = img_flat.min(dim=1, keepdim=True).values  # [B, 1]
    img_max = img_flat.max(dim=1, keepdim=True).values  # [B, 1]

    # 避免除零（当 max == min）
    denom = img_max - img_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    img_norm = (img_flat - img_min) / denom
    img_norm = img_norm.view(B, C, H, W)  # [B, 1, 512, 512]
    return img_norm

def init_config():
    return {
        "batch_size": 1,
        "lr": 5e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delx": 0.469,
        "height": 512,
        "weight_loss": 1e-2,
        "checkpoint_dir_prefix": "../output/models/loss_in_best/cube_loss",  # 保存路径前缀
        "log_dir_prefix": "../output/logs/loss_in_best/cube_loss",           # 日志路径前缀
        "log_interval": 100,
        "patience": 25,
        "min_delta": 1e-6,
        "max_saved_model_num": 5,
        "val_steps": 25,
        "max_steps": 50,
        'model_paths': {
            'mix_model': '/path/to/checkpoints/mix_model_best.pth',
            'pa_model': '/path/to/checkpoints/pa_model_best.pth',
            'rlat_model': '/path/to/checkpoints/rlat_model_best.pth'
         },
        "model_config": {
            'edffn' : 1,
            'eca' : 0,
            'conv_mlp': 0
        },
        "noise_params":{
            'trans_noise_range': [25, 25, 25],
            'rota_noise_range': [5, 5, 5]
        }
    }


# def save_results(data, prefix, save_dir, timestamp):
#     """保存数据为 CSV 和 NPY 格式"""
#
#     # 保存为 CSV
#     csv_file = os.path.join(save_dir, f"{prefix}_ZXY.csv")
#     with open(csv_file, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         # 表头 (12 列)
#         writer.writerow([
#             'pre_rota_z', 'pre_rota_x', 'pre_rota_y',
#             'pre_trans_z', 'pre_trans_x', 'pre_trans_y',
#             'tru_rota_z', 'tru_rota_x', 'tru_rota_y',
#             'tru_trans_z', 'tru_trans_x', 'tru_trans_y'
#         ])
#         # 数据行
#         writer.writerows(data)
#
#     # 保存为 NPY
#     npy_file = os.path.join(save_dir, f"{prefix}_ZXY.npy")
#     np.save(npy_file, np.array(data))
#
#     return csv_file, npy_file




def save_single_model_results(results_list: List[Dict],
                              model_name: str,
                              prefix: str,
                              save_dir: str = "./",
                              timestamp: str = None) -> Tuple[str, str]:
    """
    保存单个模型的推理结果为 CSV 和 NPY 格式

    Parameters:
    -----------
    results_list : list
        单个模型的结果列表（来自 run_single_pose 的第一个或第二个返回值）
    model_name : str
        模型名称标识（'mix_model' 或 'pa_model'/'rlat_model'）
    prefix : str
        文件名前缀（如 'model_output_pa'）
    save_dir : str
        保存目录
    timestamp : str
        时间戳

    Returns:
    --------
    tuple : (csv_path, npy_path)
    """
    # === 1. 构建文件名 ===
    csv_filename = f"{prefix}_{model_name}.csv"
    npy_filename = f"{prefix}_{model_name}.npy"


    csv_path = os.path.join(save_dir, csv_filename)
    npy_path = os.path.join(save_dir, npy_filename)

    # === 2. 转换为 DataFrame 格式 ===
    rows = []
    for batch_idx, batch_result in enumerate(results_list):
        batch_size = batch_result.get('batch_size', 1)
        pose_type = batch_result.get('pose_type', 'UNKNOWN')

        for sample_idx in range(batch_size):
            row = {
                'batch_id': batch_idx,
                'sample_id': sample_idx,
                'pose_type': pose_type,
                'model_name': model_name,

                # === 预测值（ZXY 顺序）===
                'pre_rota_z': batch_result['pre_rota'][sample_idx][0],
                'pre_rota_x': batch_result['pre_rota'][sample_idx][1],
                'pre_rota_y': batch_result['pre_rota'][sample_idx][2],
                'pre_trans_z': batch_result['pre_trans'][sample_idx][0],
                'pre_trans_x': batch_result['pre_trans'][sample_idx][1],
                'pre_trans_y': batch_result['pre_trans'][sample_idx][2],

                # === 真实值（ZXY 顺序）===
                'tru_rota_z': batch_result['tru_rota'][sample_idx][0],
                'tru_rota_x': batch_result['tru_rota'][sample_idx][1],
                'tru_rota_y': batch_result['tru_rota'][sample_idx][2],
                'tru_trans_z': batch_result['tru_trans'][sample_idx][0],
                'tru_trans_x': batch_result['tru_trans'][sample_idx][1],
                'tru_trans_y': batch_result['tru_trans'][sample_idx][2],
            }
            rows.append(row)

    # === 3. 保存 CSV ===
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"✓ [{model_name}] CSV 已保存：{csv_path} ({len(df)} 行)")

    # === 4. 保存 NPY ===
    np.save(npy_path, results_list, allow_pickle=True)
    print(f"✓ [{model_name}] NPY 已保存：{npy_path}")

    return csv_path, npy_path


def save_dual_model_results_separately(mix_results_list: List[Dict],
                                       pose_results_list: List[Dict],
                                       prefix: str,
                                       save_dir: str = "./",
                                       timestamp: str = None) -> dict:
    """
    分别保存双模型的独立结果文件

    Returns:
    --------
    dict : 包含所有保存路径的字典
    """
    # 提取模型名称
    mix_model_name = mix_results_list[0]['model_name'] if mix_results_list else 'mix_model'
    pose_model_name = pose_results_list[0]['model_name'] if pose_results_list else 'pose_model'

    # 保存 Mix Model 结果
    mix_csv, mix_npy = save_single_model_results(
        mix_results_list,
        model_name=mix_model_name,
        prefix=prefix,
        save_dir=save_dir,
        timestamp=timestamp
    )

    # 保存 Pose Model 结果
    pose_csv, pose_npy = save_single_model_results(
        pose_results_list,
        model_name=pose_model_name,
        prefix=prefix,
        save_dir=save_dir,
        timestamp=timestamp
    )

    return {
        'mix_csv': mix_csv,
        'mix_npy': mix_npy,
        'pose_csv': pose_csv,
        'pose_npy': pose_npy
    }

def run_single_pose(standard_pose: str,
                    volume_path: str,
                    global_config: dict,
                    return_dual_models: bool = True) -> Dict:
    """
    执行单个标准姿态（PA 或 RLAT）的完整推理流程，返回欧拉角。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 获取该姿态的配置
    noise_params = global_config['noise_params']
    rota_range_t = torch.tensor(
        noise_params['rota_noise_range'], dtype=torch.float32, device=device
    )
    trans_range_t = torch.tensor(
        noise_params['trans_noise_range'], dtype=torch.float32, device=device
    )
    # 2. 根据姿态类型确定模型路径
    model_paths = global_config.get('model_paths', {})

    if standard_pose.upper() == "PA":
        mix_model_path = model_paths.get('mix_model', global_config['model_path'])
        pose_model_path = model_paths.get('pa_model', global_config['model_path'])
        pose_model_name = 'pa_model'
    elif standard_pose.upper() == "RLAT":
        mix_model_path = model_paths.get('mix_model', global_config['model_path'])
        pose_model_path = model_paths.get('rlat_model', global_config['model_path'])
        pose_model_name = 'rlat_model'
    else:
        raise ValueError(f"不支持的姿态类型：{standard_pose}，仅支持 'PA' 或 'RLAT'")

    print(f"🔧 姿态类型：{standard_pose.upper()}")
    print(f"   ├─ Mix Model:   {mix_model_path}")
    print(f"   └─ Pose Model:  {pose_model_path}")

    # 3. 加载 CT
    subject = read(
        volume_path,
        bone_attenuation_multiplier=1.0,
        orientation=standard_pose,
        sid=500
    )

    # 4. 构建 DRR 渲染器
    drr = DRR(
        subject,
        sdd=800,
        height=512,
        delx=0.469,
        renderer="trilinear"
    ).to(device)

    # 5. 生成随机噪声位姿
    batch_size = global_config['batch_size']

    rot_noise_norm = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_norm = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)

    rotations = rot_noise_norm * rota_range_t  # [B, 3] (度)
    translations = trans_noise_norm * trans_range_t  # [B, 3] (mm)

    # 6. 生成 DRR
    img = drr(
        rotations, translations,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img = norm_img(img)

    # 7. 构造标签（已经是归一化的！）
    label = torch.cat([rot_noise_norm, trans_noise_norm], dim=1)  # [B, 6]

    # 8. 加载标签转换器
    label_transformer_mix = LabelTransformMix(global_config['noise_params'])

    # === 9. 🔥 双模型独立推理 ===
    mix_result = None
    pose_result = None

    with torch.no_grad():
        # --- 9.1 Mix Model 推理 ---
        print("📦 加载 Mix Model...")
        mix_model = GridModel(
            model_config=global_config['model_config'],
            num_classes=6
        ).to(device)
        mix_model.load_state_dict(torch.load(mix_model_path, map_location=device))
        mix_model.eval()

        outputs_mix = mix_model(img)
        pre_rota_mix, pre_trans_mix = label_transformer_mix.label2real(outputs_mix)
        tru_rota_mix, tru_trans_mix = label_transformer_mix.label2real(label)

        # 🔥 构建独立结果字典
        mix_result = {
            'pose_type': standard_pose.upper(),
            'batch_size': batch_size,
            'model_name': 'mix_model',
            'pre_rota': pre_rota_mix.cpu().detach().numpy().tolist(),
            'pre_trans': pre_trans_mix.cpu().detach().numpy().tolist(),
            'tru_rota': tru_rota_mix.cpu().detach().numpy().tolist(),
            'tru_trans': tru_trans_mix.cpu().detach().numpy().tolist()
        }
        # --- 9.2 Pose-Specific Model 推理 ---
        if return_dual_models:
            print(f"📦 加载 Pose Model ({pose_model_name})...")
            pose_model = GridModel(
                model_config=global_config['model_config'],
                num_classes=6
            ).to(device)
            pose_model.load_state_dict(torch.load(pose_model_path, map_location=device))
            pose_model.eval()

            outputs_pose = pose_model(img)
            pre_rota_pose, pre_trans_pose = label_transformer_mix.label2real(outputs_pose)
            tru_rota_pose, tru_trans_pose = label_transformer_mix.label2real(label)

            # 🔥 构建独立结果字典
            pose_result = {
                'pose_type': standard_pose.upper(),
                'batch_size': batch_size,
                'model_name': pose_model_name,
                'pre_rota': pre_rota_pose.cpu().detach().numpy().tolist(),
                'pre_trans': pre_trans_pose.cpu().detach().numpy().tolist(),
                'tru_rota': tru_rota_pose.cpu().detach().numpy().tolist(),
                'tru_trans': tru_trans_pose.cpu().detach().numpy().tolist()
            }

        # 清理模型释放显存
        del mix_model
        if pose_model is not None:
            del pose_model
        torch.cuda.empty_cache()

    print(f"✅ 推理完成，返回双模型独立结果")
    return mix_result, pose_result

global_config = init_config()
volume_path = r"../data/voxel_data/spine107_img.nii.gz"

# 创建保存目录
save_dir = "../data/output_results"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

mix_results_pa = []
pose_results_pa = []

mix_results_rlat = []
pose_results_rlat = []

n_iterations = 1
for i in range(n_iterations):
    try:
        # === PA 姿态推理 ===
        print(f"\n【迭代 {i + 1}/{n_iterations}】PA 姿态")
        mix_pa, pose_pa = run_single_pose(
            standard_pose="PA",
            volume_path=volume_path,
            global_config=global_config,
            return_dual_models=True
        )
        mix_results_pa.append(mix_pa)
        pose_results_pa.append(pose_pa)

        # === RLAT 姿态推理 ===
        print(f"【迭代 {i + 1}/{n_iterations}】RLAT 姿态")
        mix_rlat, pose_rlat = run_single_pose(
            standard_pose="RLAT",
            volume_path=volume_path,
            global_config=global_config,
            return_dual_models=True
        )
        mix_results_rlat.append(mix_rlat)
        pose_results_rlat.append(pose_rlat)

        # 进度显示
        if (i + 1) % 10 == 0:
            print(f"✅ 已完成 {i + 1}/{n_iterations} 次迭代")

    except Exception as e:
        print(f"❌ 第 {i + 1} 次迭代失败：{str(e)}")
        import traceback

        traceback.print_exc()
        continue

print(f"\n{'=' * 60}")
print(f"🎉 推理完成！")
print(f"{'=' * 60}")

print(f"\n📁 开始保存结果文件...")

# PA 数据保存（两个独立文件）
pa_files = save_dual_model_results_separately(
    mix_results_list=mix_results_pa,
    pose_results_list=pose_results_pa,
    prefix="model_output_pa",
    save_dir=save_dir,
    timestamp=timestamp
)

# RLAT 数据保存（两个独立文件）
rlat_files = save_dual_model_results_separately(
    mix_results_list=mix_results_rlat,
    pose_results_list=pose_results_rlat,
    prefix="model_output_rlat",
    save_dir=save_dir,
    timestamp=timestamp
)

# ================= 5. 打印保存信息 =================
print(f"\n{'='*60}")
print(f"📊 数据保存完成！")
print(f"{'='*60}")
print(f"\n📁 PA 姿态输出文件:")
print(f"  Mix Model CSV:  {pa_files['mix_csv']}")
print(f"  Mix Model NPY:  {pa_files['mix_npy']}")
print(f"  Pose Model CSV: {pa_files['pose_csv']}")
print(f"  Pose Model NPY: {pa_files['pose_npy']}")

print(f"\n📁 RLAT 姿态输出文件:")
print(f"  Mix Model CSV:  {rlat_files['mix_csv']}")
print(f"  Mix Model NPY:  {rlat_files['mix_npy']}")
print(f"  Pose Model CSV: {rlat_files['pose_csv']}")
print(f"  Pose Model NPY: {rlat_files['pose_npy']}")

print(f"\n📂 保存目录：{os.path.abspath(save_dir)}")
print(f"{'='*60}\n")




