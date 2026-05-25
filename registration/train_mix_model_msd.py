import csv
import time
from pathlib import Path
import nibabel as nib

import numpy as np
import os
import json
import torch
import torch.optim as optim
from click.core import F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from grid_efficientnet.grid_efficientb0_model import GridModel

from projector.drr import DRR
from projector.read_data import read

from label_transform_mix import LabelTransformMix
from label_transform_mix_2 import LabelTransformMix2
from math_process.get_noise_label import get_transformer_noise, get_transformer_noise_vector

from datetime import datetime

from registration.loss_model_msd import PoseMSDPredictor
from registration.projector.pose import convert


def init_noise_logger(base_dir: str):
    """初始化噪声日志：返回保存函数 + 关闭函数"""
    log_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, "noise_logs")
    os.makedirs(log_dir, exist_ok=True)

    # 打开两个CSV文件（带缓冲，提升写入效率）
    files = {
        'pa': open(os.path.join(log_dir, f"{log_id}_pa_noise.csv"), 'w', newline='', encoding='utf-8'),
        'rlat': open(os.path.join(log_dir, f"{log_id}_rlat_noise.csv"), 'w', newline='', encoding='utf-8')
    }
    # 写入表头
    header = ['step', 'batch_idx', 'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    for f in files.values():
        csv.writer(f).writerow(header)

    def save(step: int, rot: torch.Tensor, trans: torch.Tensor, view: str):
        """保存单步噪声：rot/trans shape=[batch, 3]"""
        if view not in files: return
        writer = csv.writer(files[view])
        rot_np, trans_np = rot.cpu().numpy(), trans.cpu().numpy()
        for idx in range(rot_np.shape[0]):
            writer.writerow([step, idx, *rot_np[idx], *trans_np[idx]])

    def close():
        """训练结束：关闭文件"""
        for f in files.values(): f.close()
        print(f"✅ Noise logs saved: {log_dir}")

    return {'save': save, 'close': close, 'log_id': log_id}


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

def norm_label(rot_noise, trans_noise):
    """
    噪声归一化
    输入:
        rot_noise: (B, 3) tensor, 度
        trans_noise: (B, 3) tensor, 单位 (mm/m)
    输出:
        rot_norm: (B, 3) tensor, 归一化后
        trans_norm: (B, 3) tensor, 归一化后
    """
    device = rot_noise.device
    rot_coef = rot_norm_coef.to(device)
    trans_coef = trans_norm_coef.to(device)

    rot_norm = rot_noise / rot_coef
    trans_norm = trans_noise / trans_coef

    return rot_norm, trans_norm



def init_config():
    return {
        "batch_size": 4,
        "lr": 5e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delx": 0.469,
        "height": 512,
        "weight_loss": 1e-2,
        "checkpoint_dir_prefix": "../output/models/loss_in_best/cube_loss",  # 保存路径前缀
        "log_dir_prefix": "../output/logs/loss_in_best/cube_loss",  # 日志路径前缀
        "loss_model": '../data/loss_data/loss_models/model_v4/best_pose_msd.pth',
        "log_interval": 100,
        "patience": 100,
        "min_lr": 1e-5,
        "max_saved_model_num": 5,
        "val_steps": 25,
        "max_steps": 20000,
        "R_ct2osz":[[-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]],
        "R_ct2osc":[[0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1]],
        "model_config": {
            'edffn': 1,
            'eca': 0,
            'conv_mlp': 0
        },
        "noise_params": {
            'trans_noise_range': [25, 25, 25],
            'rota_noise_range': [5, 5, 5]
        },
        "norm_params":{
            'trans_noise_norm':[25, 25, 25],
            'rota_noise_norm':[10, 10, 10]
        }
    }


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


def compute_voxel_msd_params(vol_path):
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

    return mu_p, Sigma_p


def get_msd_loss(config, label_transformer_mix, labels, outputs, mu_p, Sigma_p):
    """
    基于闭式解的位姿差异 MSD Loss (完全可微，O(1) 复杂度)
    公式: MSD = tr(ΔRᵀΔR · Σ_p) + ‖ΔR·μ_p + Δt‖²
    """
    device = outputs.device

    # 1️⃣ 确保统计量在正确设备与类型
    mu_p = torch.as_tensor(mu_p, dtype=torch.float32, device=device)  # [3] 或 [B, 3]
    Sigma_p = torch.as_tensor(Sigma_p, dtype=torch.float32, device=device)  # [3, 3]

    # 2️⃣ 解码位姿参数
    pre_rota, pre_trans = label_transformer_mix.label2real(outputs)
    tru_rota, tru_trans = label_transformer_mix.label2real(labels)

    # 转换得到 pose 对象 (假设返回 Batch 张量)
    tru_pose = convert(tru_rota, tru_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
    pre_pose = convert(pre_rota, pre_trans, parameterization="euler_angles", convention="ZXY", degrees=True)

    # 3️⃣ 提取旋转矩阵与平移向量 (保持与原代码一致的 .mT 转置)
    tru_R = tru_pose.rotation.mT.to(device)  # [B, 3, 3]
    pre_R = pre_pose.rotation.mT.to(device)  # [B, 3, 3]
    tru_t = tru_pose.translation.to(device)  # [B, 3]
    pre_t = pre_pose.translation.to(device)  # [B, 3]

    # 4️⃣ 计算位姿差 ΔR, Δt
    dR = tru_R - pre_R  # [B, 3, 3]
    dt = tru_t - pre_t  # [B, 3]

    # 5️⃣ 闭式解 MSD 计算 (核心替换)
    # 🔹 Term 1: tr(ΔRᵀΔR · Σ_p)
    A = torch.matmul(dR.transpose(-2, -1), dR)  # [B, 3, 3]
    # 利用迹的性质 tr(A·Σ) = Σ(A * Σ) (Σ_p 为对称协方差矩阵)
    term1 = (A * Sigma_p.unsqueeze(0)).sum(dim=(-2, -1))  # [B]

    # 🔹 Term 2: ‖ΔR·μ_p + Δt‖²
    center_disp = torch.matmul(dR, mu_p) + dt  # [B, 3] (自动广播批处理)
    term2 = torch.sum(center_disp ** 2, dim=-1)  # [B]

    # 合成 MSD [B]
    msd = term1 + term2  # [B]

    # 6️⃣ 计算最终 Loss (取批次均值)
    loss = config.get("weight_loss", 1.0) * msd.mean()
    return loss

global_config = init_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_params = global_config['noise_params']
rota_range_t = torch.tensor(
    noise_params['rota_noise_range'], dtype=torch.float32, device=device
)
trans_range_t = torch.tensor(
    noise_params['trans_noise_range'], dtype=torch.float32, device=device
)
norm_params = global_config['norm_params']
rot_norm_coef = torch.tensor(norm_params['rota_noise_norm'], dtype=torch.float32, device=device)
trans_norm_coef = torch.tensor(norm_params['trans_noise_norm'], dtype=torch.float32, device=device)
R_ct2osz = torch.tensor(global_config['R_ct2osz'], dtype=torch.float32, device=device)
R_ct2osc = torch.tensor(global_config['R_ct2osc'], dtype=torch.float32, device=device)

patience = global_config['patience']
min_lr = global_config['min_lr']
height = global_config['height']
delx = global_config['delx']
batch_size = global_config['batch_size']

label_transformer_mix1 = LabelTransformMix(global_config['noise_params'])
label_transformer_mix2 = LabelTransformMix2(global_config['norm_params'])
volume_dir_2 = r"../data/voxel_data/spine107_img.nii.gz"

mu_p, Sigma_p = compute_voxel_msd_params(volume_dir_2)

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

# mix1模型
model_mix1 = GridModel(model_config=global_config['model_config'], num_classes=6).to(device)
optimizer_mix1 = optim.Adam(model_mix1.parameters(), lr=global_config["lr"])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
scheduler_mix1 = ReduceLROnPlateau(
    optimizer_mix1,
    mode='min',
    factor=0.5,
    patience=patience,  # ← 从 1 改为 200，给模型更多收敛时间
    min_lr=min_lr,  # ← 设置下限，防止过小
)

# mix2模型
model_mix2 = GridModel(model_config=global_config['model_config'], num_classes=6).to(device)
optimizer_mix2 = optim.Adam(model_mix2.parameters(), lr=global_config["lr"])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
scheduler_mix2 = ReduceLROnPlateau(
    optimizer_mix2,
    mode='min',
    factor=0.5,
    patience=patience,  # ← 从 1 改为 200，给模型更多收敛时间
    min_lr=min_lr,  # ← 设置下限，防止过小
)

# === 统一配置保存路径 ===
base_checkpoint_dir = "uliver4_model"
os.makedirs(base_checkpoint_dir, exist_ok=True)

noise_ctx = init_noise_logger(base_checkpoint_dir)

# ✅ 修复：两个模型使用独立子目录
checkpoint_dir_mix1 = os.path.join(base_checkpoint_dir, "mix1_view")
checkpoint_dir_mix2 = os.path.join(base_checkpoint_dir, "mix2_view")
os.makedirs(checkpoint_dir_mix1, exist_ok=True)
os.makedirs(checkpoint_dir_mix2, exist_ok=True)

# === 初始化训练状态 ===
max_steps = global_config['max_steps']
# pts = sample_cube_points(3).to(device)

# 确保 batch_size 是偶数
assert batch_size % 2 == 0, "Batch size must be even to split 50/50"
half_batch = batch_size // 2


# 📈 分别追踪两个模型的损失和最佳状态
training_state = {
    "mix1": {
        "losses": [],
        "best_loss": float('inf'),
        "best_step": 0,
        "model": model_mix1,
        "optimizer": optimizer_mix1,
        "scheduler": scheduler_mix1,
        "label_transformer": label_transformer_mix1,
        "checkpoint_dir": checkpoint_dir_mix1
    },
    "mix2": {
        "losses": [],
        "best_loss": float('inf'),
        "best_step": 0,
        "model": model_mix2,
        "optimizer": optimizer_mix2,
        "scheduler": scheduler_mix2,
        "label_transformer": label_transformer_mix2,
        "checkpoint_dir": checkpoint_dir_mix2
    }
}

log_interval = global_config.get('log_interval', 100)
save_interval = global_config.get('save_interval', 1000)

for step in range(max_steps):

    # --- PA 视角部分 ---
    rot_noise_pa = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_pa = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)

    noise_ctx['save'](step + 1, rot_noise_pa, trans_noise_pa, 'pa')

    rotations_pa = rot_noise_pa * rota_range_t
    translations_pa = trans_noise_pa * trans_range_t

    rot_pa,trans_pa = get_transformer_noise(rotations_pa, translations_pa, R_ct2osz)
    label_rot_pa, label_trans_pa = norm_label(rot_pa, trans_pa)

    img_pa = drr_pa(
        rotations_pa,
        translations_pa,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img_pa = norm_img(img_pa)
    label_pa_mix1 = torch.cat([rot_noise_pa, trans_noise_pa], dim=1)
    label_pa_mix2 = torch.cat([label_rot_pa,label_trans_pa], dim=1)  # [8, 6]

    # --- RLAT 视角部分 ---
    rot_noise_rlat = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_rlat = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)

    noise_ctx['save'](step + 1, rot_noise_rlat, trans_noise_rlat, 'rlat')

    rotations_rlat = rot_noise_rlat * rota_range_t
    translations_rlat = trans_noise_rlat * trans_range_t

    rot_rlat, trans_rlat = get_transformer_noise(rotations_rlat, translations_rlat, R_ct2osc)
    label_rot_rlat, label_trans_rlat = norm_label(rot_rlat, trans_rlat)

    img_rlat = drr_rlat(
        rotations_rlat,
        translations_rlat,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img_rlat = norm_img(img_rlat)
    label_rlat_mix1 = torch.cat([rot_noise_rlat, trans_noise_rlat], dim=1)
    label_rlat_mix2 = torch.cat([label_rot_rlat, label_trans_rlat], dim=1)  # [8, 6]


    # --- 混合模型数据：拼接 + 打乱 ---
    img_mix = torch.cat([img_pa, img_rlat], dim=0)  # [16, H, W]

    label_mix1 = torch.cat([label_pa_mix1, label_rlat_mix1], dim=0)  # [16, 6]
    label_mix2 = torch.cat([label_pa_mix2, label_rlat_mix2], dim=0)

    shuffle_idx = torch.randperm(batch_size, device=device)
    img_mix = img_mix[shuffle_idx]

    label_mix1 = label_mix1[shuffle_idx]
    label_mix2 = label_mix2[shuffle_idx]

    # ================= 模型 1: 混合视角训练 =================
    model_mix1.train()
    output_mix1 = model_mix1(img_mix)
    loss_mix1 = get_msd_loss(global_config, label_transformer_mix1, label_mix1, output_mix1, mu_p, Sigma_p)
    loss_mix1_val = loss_mix1.item()

    optimizer_mix1.zero_grad()
    loss_mix1.backward()
    torch.nn.utils.clip_grad_norm_(model_mix1.parameters(), max_norm=1.0)
    optimizer_mix1.step()
    scheduler_mix1.step(loss_mix1_val)
    training_state["mix1"]["losses"].append(loss_mix1_val)

    # ================= 模型 2: 混合视角训练 =================
    model_mix2.train()
    output_mix2 = model_mix2(img_mix)
    loss_mix2 = get_msd_loss(global_config, label_transformer_mix2, label_mix2, output_mix2, mu_p, Sigma_p)
    loss_mix2_val = loss_mix2.item()

    optimizer_mix2.zero_grad()
    loss_mix2.backward()
    torch.nn.utils.clip_grad_norm_(model_mix2.parameters(), max_norm=1.0)
    optimizer_mix2.step()
    scheduler_mix2.step(loss_mix2_val)
    training_state["mix2"]["losses"].append(loss_mix2_val)

    # ================= 定期输出训练指标 + 记录详细日志 =================
    if (step + 1) % log_interval == 0 or (step + 1) == max_steps:
        current_time = time.time()  # 🔹 新增：时间戳用于计算耗时

        print(f"\n{'=' * 70}")
        print(f"Step [{step + 1:6d}/{max_steps}]")
        print(f"{'-' * 70}")

        for name, state in training_state.items():
            with torch.no_grad():
                model = state["model"]
                output = output_mix1 if name == "mix1" else output_mix2
                label = label_mix1 if name == "mix1" else label_mix2

                out_range = f"[{output.min().item():.3f}, {output.max().item():.3f}]"
                lbl_range = f"[{label.min().item():.3f}, {label.max().item():.3f}]"

                # 安全获取梯度范数
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad).item()
                        break

                lr = state["optimizer"].param_groups[0]['lr']
                current_loss = state["losses"][-1]

            # 🔹 打印原有信息（保持不变）
            print(f" [{name.upper()}] Loss: {current_loss:.6f} | LR: {lr:.2e} | "
                  f"Out: {out_range} | Grad: {grad_norm:.4f}")

            # 🔹 新增：构建详细日志条目
            log_entry = {
                "step": step + 1,
                "timestamp": current_time,
                "loss": float(current_loss),
                "lr": float(lr),
                "grad_norm": float(grad_norm),
                "output_range": [float(output.min().item()), float(output.max().item())],
                "label_range": [float(label.min().item()), float(label.max().item())],
                "is_best": current_loss < state["best_loss"]  # 标记是否为当前最佳
            }

            # 🔹 新增：追加写入 JSONL 文件（每个模型独立）
            log_path = os.path.join(state["checkpoint_dir"], "training_log.jsonl")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"{'=' * 70}\n")

    # ================= 保存最佳模型（双模型独立） =================
    for name, state in training_state.items():
        current_loss = state["losses"][-1]
        if current_loss < state["best_loss"]:
            improvement = (state["best_loss"] - current_loss) / state["best_loss"] * 100 if state[
                                                                                                "best_loss"] != float(
                'inf') else 0
            state["best_loss"] = current_loss
            state["best_step"] = step + 1

            # 保存完整检查点
            checkpoint = {
                'step': step + 1,
                'model_state_dict': state["model"].state_dict(),
                'optimizer_state_dict': state["optimizer"].state_dict(),
                'scheduler_state_dict': state["scheduler"].state_dict(),
                'loss': current_loss,
                'config': global_config,
                'label_transformer_type': 'mix1' if name == "mix1" else 'mix2'
            }
            save_path = os.path.join(state["checkpoint_dir"], "best_model.pth")
            torch.save(checkpoint, save_path)

            # 🎉 输出最佳迭代信息
            with torch.no_grad():
                output = output_mix1 if name == "mix1" else output_mix2
                label = label_mix1 if name == "mix1" else label_mix2
                out_range = f"[{output.min().item():.4f}, {output.max().item():.4f}]"
                lbl_range = f"[{label.min().item():.4f}, {label.max().item():.4f}]"

            print(f"[{name.upper()}] 🏆 NEW BEST at Step {step + 1}!")
            print(f"   Loss: {current_loss:.8f} (↓{improvement:.2f}% vs prev best)")
            print(f"   Output: {out_range} | Label: {lbl_range}")
            print(f"   Saved to: {save_path}\n")

    # ================= 定期保存中间检查点 =================
    if (step + 1) % save_interval == 0 and (step + 1) < max_steps:
        for name, state in training_state.items():
            save_path = os.path.join(state["checkpoint_dir"], f"model_step_{step + 1}.pth")
            torch.save(state["model"].state_dict(), save_path)
            print(f"💾 [{name.upper()}] Checkpoint saved at step {step + 1}: {save_path}")

            print("\n🏁 Training finished. Saving final models and logs...")

# --- 保存最终模型权重 + 损失日志 ---
for name, state in training_state.items():
    # 保存最终模型
    final_path = os.path.join(state["checkpoint_dir"], "final_model.pth")
    torch.save(state["model"].state_dict(), final_path)

    # 保存 losses 日志 (JSON + TXT)
    losses = state["losses"]
    steps = list(range(1, len(losses) + 1))

    # JSON 格式（便于 Python 读取绘图）
    with open(os.path.join(state["checkpoint_dir"], "losses.json"), "w", encoding='utf-8') as f:
        json.dump({"steps": steps, "losses": losses}, f, indent=2, ensure_ascii=False)

    # TXT 格式（便于快速查看）
    with open(os.path.join(state["checkpoint_dir"], "losses.txt"), "w", encoding='utf-8') as f:
        f.write("step,loss\n")
        for s, loss_val in zip(steps, losses):
            f.write(f"{s},{loss_val:.8f}\n")

    print(f"📊 [{name.upper()}] Logs saved to {state['checkpoint_dir']}")

# --- 保存训练汇总信息（统一摘要） ---
summary = {
    "total_steps": max_steps,
    "batch_size": batch_size,
    "half_batch": half_batch,
    "device": str(device),
    "best_results": {
        name: {
            "best_loss": float(state["best_loss"]),
            "best_step": state["best_step"],
            "final_loss": float(state["losses"][-1]) if state["losses"] else None,
            "checkpoint_dir": state["checkpoint_dir"]
        }
        for name, state in training_state.items()
    },
    "config_snapshot": {
        "lr": global_config.get("lr"),
        "patience": global_config.get("patience"),
        "min_lr": global_config.get("min_lr"),
        "noise_params": global_config.get("noise_params"),
        "norm_params": global_config.get("norm_params")
    }
}

summary_path = os.path.join(base_checkpoint_dir, "training_summary.json")
with open(summary_path, "w", encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# --- 输出目录结构 ---
noise_ctx['close']()
print(f"\n✅ All models and logs saved to: {base_checkpoint_dir}")
print(f"📁 Directory structure:")
print(f"   {base_checkpoint_dir}/")
for name, state in training_state.items():
    print(f"   ├── {os.path.basename(state['checkpoint_dir'])}/")
    print(f"   │   ├── best_model.pth      ← 最佳验证损失模型")
    print(f"   │   ├── final_model.pth     ← 训练结束时的模型")
    print(f"   │   ├── losses.json         ← 损失曲线数据（JSON）")
    print(f"   │   └── losses.txt          ← 损失曲线数据（CSV）")
print(f"   └── training_summary.json  ← 双模型训练汇总")










