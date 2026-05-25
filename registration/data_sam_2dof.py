import os
import pandas as pd
from typing import Dict, List, Tuple

import numpy as np
import torch

from grid_efficientnet.grid_efficientb0_model import GridModel

from label_transform_mix_2 import LabelTransformMix2
from math_process.get_noise_label import get_transformer_noise
from projector.drr import DRR
from projector.read_data import read

from label_transform_mix import LabelTransformMix


def init_config():
    return {
        "batch_size": 1,
        "lr": 5e-4,
        "min_lr": 1e-6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delx": 0.469,
        "height": 512,
        "weight_loss": 1e-2,
        "checkpoint_dir_prefix": "../output/models/loss_in_best/cube_loss",  # 保存路径前缀
        "log_dir_prefix": "../output/logs/loss_in_best/cube_loss",  # 日志路径前缀
        "log_interval": 100,
        "patience": 25,
        "min_delta": 1e-6,
        "max_saved_model_num": 5,
        "val_steps": 25,
        "max_steps": 50,
        'mix1_model_path': 'data/reg_model/mix1_uliver6_2dof.pth',
        'mix2_model_path': 'data/reg_model/mix2_uliver6_2dof.pth',
        'mix3_model_path': 'data/reg_model/mix3_uliver6_model.pth',
        "model_config": {
            'edffn': 1,
            'eca': 0,
            'conv_mlp': 0
        },
        "R_ct2osz": [[-1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]],
        "R_ct2osc": [[0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [-1, 0, 0, 0],
                     [0, 0, 0, 1]],
        "noise_params": {
            'trans_noise_range': [25, 25, 0],
            'rota_noise_range': [0, 0, 0]
        },
        "norm_params": {
            'trans_noise_range': [25, 25, 25],
            'rota_noise_range': [1, 1, 1]
        }
    }


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


def norm_label(rot_noise, trans_noise, rot_coef, trans_coef):
    """
    修正：显式传入系数和设备，不再依赖全局变量
    """
    rot_norm = rot_noise / rot_coef
    trans_norm = trans_noise / trans_coef
    return rot_norm, trans_norm


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
    # npy_filename = f"{prefix}_{model_name}.npy"

    csv_path = os.path.join(save_dir, csv_filename)
    # npy_path = os.path.join(save_dir, npy_filename)

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
                'pre_trans_x': batch_result['pre_trans'][sample_idx][0],
                'pre_trans_y': batch_result['pre_trans'][sample_idx][1],
                'pre_trans_z': batch_result['pre_trans'][sample_idx][2],

                # === 真实值（ZXY 顺序）===
                'tru_rota_z': batch_result['tru_rota'][sample_idx][0],
                'tru_rota_x': batch_result['tru_rota'][sample_idx][1],
                'tru_rota_y': batch_result['tru_rota'][sample_idx][2],
                'tru_trans_x': batch_result['tru_trans'][sample_idx][0],
                'tru_trans_y': batch_result['tru_trans'][sample_idx][1],
                'tru_trans_z': batch_result['tru_trans'][sample_idx][2],
            }
            rows.append(row)

    # === 3. 保存 CSV ===
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"✓ [{model_name}] CSV 已保存：{csv_path} ({len(df)} 行)")

    # # === 4. 保存 NPY ===
    # np.save(npy_path, results_list, allow_pickle=True)
    # print(f"✓ [{model_name}] NPY 已保存：{npy_path}")

    return csv_path


def save_results_wrapper(results_list: List[Dict], prefix: str, save_dir: str) -> dict:
    """
    修正：根据实际模型名称动态返回键名
    """
    if not results_list:
        return {}

    model_name = results_list[0].get('model_name', 'unknown_model')

    csv_path = save_single_model_results(
        results_list, model_name=model_name, prefix=prefix, save_dir=save_dir
    )

    # 动态生成键名，例如 mix1_csv, mix2_csv
    key_prefix = model_name.replace('_model', '')
    return {
        f'{key_prefix}_csv': csv_path
        # f'{key_prefix}_npy': npy_path
    }


if __name__ == "__main__":
    # 1. 初始化配置
    global_config = init_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume_dir_2 = r"data/voxel_data/uniformed_liver_6.nii.gz"
    height = global_config['height']
    delx = global_config['delx']
    batch_size = global_config['batch_size']

    # 2. 准备归一化系数 (提前转到 device)
    noise_params = global_config['noise_params']
    rota_range_t = torch.tensor(
        noise_params['rota_noise_range'], dtype=torch.float32, device=device
    )
    trans_range_t = torch.tensor(
        noise_params['trans_noise_range'], dtype=torch.float32, device=device
    )
    norm_params = global_config['norm_params']
    rot_coef = torch.tensor(norm_params['rota_noise_norm'], dtype=torch.float32, device=device)
    trans_coef = torch.tensor(norm_params['trans_noise_norm'], dtype=torch.float32, device=device)

    R_ct2osz = torch.tensor(global_config['R_ct2osz'], dtype=torch.float32, device=device)
    R_ct2osc = torch.tensor(global_config['R_ct2osc'], dtype=torch.float32, device=device)

    # 加载 CT 和渲染
    subject_pa = read(volume_dir_2, bone_attenuation_multiplier=1.0, orientation='PA', sid=500)
    subject_rlat = read(volume_dir_2, bone_attenuation_multiplier=1.0, orientation='RLAT', sid=500)

    drr_pa = DRR(
        subject_pa,  # An object storing the CT volume, origin, and voxel spacing
        sdd=800,  # Source-to-detector distance (i.e., focal length)
        height=height,  # Image height (if width is not provided, the generated DRR is square)
        delx=delx,  # Pixel spacing (in mm)
        renderer="trilinear"
    ).to(device)
    drr_rlat = DRR(
        subject_rlat,  # An object storing the CT volume, origin, and voxel spacing
        sdd=800,  # Source-to-detector distance (i.e., focal length)
        height=height,  # Image height (if width is not provided, the generated DRR is square)
        delx=delx,  # Pixel spacing (in mm)
        renderer="trilinear"
    ).to(device)



    # 3. 【关键修正】预加载模型 (只在循环外加载一次)
    print("⏳ 正在加载模型...")
    mix1_model = GridModel(model_config=global_config['model_config'], num_classes=6).to(device)
    mix1_model.load_state_dict(torch.load(global_config['mix1_model_path'], map_location=device))
    mix1_model.eval()

    mix2_model = GridModel(model_config=global_config['model_config'], num_classes=6).to(device)
    mix2_model.load_state_dict(torch.load(global_config['mix2_model_path'], map_location=device))
    mix2_model.eval()


    print("✅ 模型加载完成")

    # 4. 准备标签转换器
    label_transformer_mix = LabelTransformMix2(global_config['norm_params'])

    # 5. 准备保存目录
    save_dir = "data/uliver6_2dof_data"
    os.makedirs(save_dir, exist_ok=True)

    # 6. 结果收集列表
    mix1_results_pa, mix1_results_rlat = [], []
    mix2_results_pa, mix2_results_rlat = [], []


    n_iterations = 5000
    print(f"🚀 开始推理，共 {n_iterations} 次迭代...")
    for i in range(n_iterations):
        # --- PA 视角部分 ---
        rot_noise_pa = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
        trans_noise_pa = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)

        rotations_pa = rot_noise_pa * rota_range_t
        translations_pa = trans_noise_pa * trans_range_t

        rot_pa, trans_pa = get_transformer_noise(rotations_pa, translations_pa, R_ct2osz)
        label_rot_pa, label_trans_pa = norm_label(rot_pa, trans_pa, rot_coef, trans_coef)

        img_pa = drr_pa(
            rotations_pa,
            translations_pa,
            parameterization="euler_angles", convention="ZXY", degrees=True
        )
        img_pa = norm_img(img_pa)
        label_pa_mix1 = torch.cat([rot_noise_pa, trans_noise_pa], dim=1)
        label_pa_mix2 = torch.cat([label_rot_pa, label_trans_pa], dim=1)


        # --- RLAT 视角部分 ---
        rot_noise_rlat = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)
        trans_noise_rlat = torch.empty(batch_size, 3, device=device).uniform_(-1.0, 1.0)

        rotations_rlat = rot_noise_rlat * rota_range_t
        translations_rlat = trans_noise_rlat * trans_range_t

        rot_rlat, trans_rlat = get_transformer_noise(rotations_rlat, translations_rlat, R_ct2osc)
        label_rot_rlat, label_trans_rlat = norm_label(rot_rlat, trans_rlat, rot_coef, trans_coef)

        img_rlat = drr_rlat(
            rotations_rlat,
            translations_rlat,
            parameterization="euler_angles", convention="ZXY", degrees=True
        )
        img_rlat = norm_img(img_rlat)
        label_rlat_mix1 = torch.cat([rot_noise_rlat, trans_noise_rlat], dim=1)
        label_rlat_mix2 = torch.cat([label_rot_rlat, label_trans_rlat], dim=1)



        # --- 推理 ---
        with torch.no_grad():
            # Mix1
            outputs_pa_mix1 = mix1_model(img_pa)
            pre_rota_pa_mix1, pre_trans_pa_mix1 = label_transformer_mix.label2real(outputs_pa_mix1)
            tru_rota_pa_mix1, tru_trans_pa_mix1 = label_transformer_mix.label2real(label_pa_mix1)

            outputs_rlat_mix1 = mix1_model(img_rlat)
            pre_rota_rlat_mix1, pre_trans_rlat_mix1 = label_transformer_mix.label2real(outputs_rlat_mix1)
            tru_rota_rlat_mix1, tru_trans_rlat_mix1 = label_transformer_mix.label2real(label_rlat_mix1)

            # Mix2
            outputs_pa_mix2 = mix2_model(img_pa)
            pre_rota_pa_mix2, pre_trans_pa_mix2 = label_transformer_mix.label2real(outputs_pa_mix2)
            tru_rota_pa_mix2, tru_trans_pa_mix2 = label_transformer_mix.label2real(label_pa_mix2)

            outputs_rlat_mix2 = mix2_model(img_rlat)
            pre_rota_rlat_mix2, pre_trans_rlat_mix2 = label_transformer_mix.label2real(outputs_rlat_mix2)
            tru_rota_rlat_mix2, tru_trans_rlat_mix2 = label_transformer_mix.label2real(label_rlat_mix2)


        # --- 构建结果字典 ---
        mix1_pa_result = {
            'pose_type': 'PA', 'batch_size': batch_size, 'model_name': 'mix1_model',
            'pre_rota': pre_rota_pa_mix1.cpu().detach().numpy().tolist(),
            'pre_trans': pre_trans_pa_mix1.cpu().detach().numpy().tolist(),
            'tru_rota': tru_rota_pa_mix1.cpu().detach().numpy().tolist(),
            'tru_trans': tru_trans_pa_mix1.cpu().detach().numpy().tolist()
        }
        mix1_rlat_result = {
            'pose_type': 'RLAT', 'batch_size': batch_size, 'model_name': 'mix1_model',
            'pre_rota': pre_rota_rlat_mix1.cpu().detach().numpy().tolist(),
            'pre_trans': pre_trans_rlat_mix1.cpu().detach().numpy().tolist(),
            'tru_rota': tru_rota_rlat_mix1.cpu().detach().numpy().tolist(),
            'tru_trans': tru_trans_rlat_mix1.cpu().detach().numpy().tolist()
        }
        mix2_pa_result = {
            'pose_type': 'PA', 'batch_size': batch_size, 'model_name': 'mix2_model',
            'pre_rota': pre_rota_pa_mix2.cpu().detach().numpy().tolist(),
            'pre_trans': pre_trans_pa_mix2.cpu().detach().numpy().tolist(),
            'tru_rota': tru_rota_pa_mix2.cpu().detach().numpy().tolist(),
            'tru_trans': tru_trans_pa_mix2.cpu().detach().numpy().tolist()
        }
        mix2_rlat_result = {
            'pose_type': 'RLAT', 'batch_size': batch_size, 'model_name': 'mix2_model',
            'pre_rota': pre_rota_rlat_mix2.cpu().detach().numpy().tolist(),
            'pre_trans': pre_trans_rlat_mix2.cpu().detach().numpy().tolist(),
            'tru_rota': tru_rota_rlat_mix2.cpu().detach().numpy().tolist(),
            'tru_trans': tru_trans_rlat_mix2.cpu().detach().numpy().tolist()
        }


        mix1_results_pa.append(mix1_pa_result)
        mix1_results_rlat.append(mix1_rlat_result)
        mix2_results_pa.append(mix2_pa_result)
        mix2_results_rlat.append(mix2_rlat_result)


        if (i + 1) % 50 == 0:
            print(f"✅ 进度：{i + 1}/{n_iterations}")

    print(f"\n🎉 推理完成！开始保存...")

    # 7. 【关键修正】保存并使用正确的变量名
    # PA
    mix1_pa_files = save_results_wrapper(mix1_results_pa, f"{n_iterations}_copy2_pa", save_dir)
    mix2_pa_files = save_results_wrapper(mix2_results_pa, f"{n_iterations}_copy2_pa", save_dir)


    # RLAT
    mix1_rlat_files = save_results_wrapper(mix1_results_rlat, f"{n_iterations}_copy2_rlat", save_dir)
    mix2_rlat_files = save_results_wrapper(mix2_results_rlat, f"{n_iterations}_copy2_rlat", save_dir)


    # 8. 打印信息 (使用正确的变量，并补充 Mix1 的信息)
    print(f"\n{'=' * 60}")
    print(f"📊 数据保存完成！")
    print(f"{'=' * 60}")

    # === PA 姿态输出文件 ===
    print(f"\n📁 PA 姿态输出文件:")
    if mix1_pa_files:
        print(f"  Mix1 CSV:  {mix1_pa_files.get('mix1_csv', 'N/A')}")
        # print(f"  Mix1 NPY:  {mix1_pa_files.get('mix1_npy', 'N/A')}")
    if mix2_pa_files:
        print(f"  Mix2 CSV:  {mix2_pa_files.get('mix2_csv', 'N/A')}")
        # print(f"  Mix2 NPY:  {mix2_pa_files.get('mix2_npy', 'N/A')}")


    # === RLAT 姿态输出文件 ===
    print(f"\n📁 RLAT 姿态输出文件:")
    if mix1_rlat_files:
        print(f"  Mix1 CSV:  {mix1_rlat_files.get('mix1_csv', 'N/A')}")
        # print(f"  Mix1 NPY:  {mix1_rlat_files.get('mix1_npy', 'N/A')}")
    if mix2_rlat_files:
        print(f"  Mix2 CSV:  {mix2_rlat_files.get('mix2_csv', 'N/A')}")
        # print(f"  Mix2 NPY:  {mix2_rlat_files.get('mix2_npy', 'N/A')}")


    print(f"\n📂 保存目录：{os.path.abspath(save_dir)}")
    print(f"{'=' * 60}\n")

    # 清理显存
    del mix1_model, mix2_model
    torch.cuda.empty_cache()