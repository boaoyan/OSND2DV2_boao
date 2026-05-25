import numpy as np
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from grid_efficientnet.grid_efficientb0_model import GridModel

from projector.drr import DRR
from projector.read_data import read
from projector.pose import convert
from label_transform_mix_2 import LabelTransformMix2
from math_process.get_noise_label import get_transformer_noise, get_transformer_noise_vector


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
        "batch_size": 16,
        "lr": 5e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delx": 0.469,
        "height": 512,
        "weight_loss": 1e-2,
        "checkpoint_dir_prefix": "../output/models/loss_in_best/cube_loss",  # 保存路径前缀
        "log_dir_prefix": "../output/logs/loss_in_best/cube_loss",  # 日志路径前缀
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


def get_mse_loss(config, label_transformer_mix, labels, outputs, pts):
    # 1️⃣ 获取 pts 的设备（作为基准设备）
    device = outputs.device  # ← 以模型输出为基准（通常在 GPU）
    pts = pts.to(device)  # ← 确保 pts 在同一设备

    pre_rota, pre_trans = label_transformer_mix.label2real(outputs)
    tru_rota, tru_trans = label_transformer_mix.label2real(labels)

    # 2️⃣ 转换得到 pose 对象
    tru_pose = convert(tru_rota, tru_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
    pre_pose = convert(pre_rota, pre_trans, parameterization="euler_angles", convention="ZXY", degrees=True)

    # 3️⃣ 🔥 关键修复：将 rotation 和 translation 迁移到 pts 同一设备
    tru_rotation = tru_pose.rotation.mT.to(device)
    pre_rotation = pre_pose.rotation.mT.to(device)
    tru_translation = tru_pose.translation.to(device)
    pre_translation = pre_pose.translation.to(device)

    # 4️⃣ 执行矩阵运算（现在所有张量都在同一设备）
    tru_pts = torch.matmul(tru_rotation, pts.T)
    pre_pts = torch.matmul(pre_rotation, pts.T)

    # 5️⃣ 计算损失
    trans_loss = F.mse_loss(tru_translation, pre_translation)
    loss = F.mse_loss(tru_pts, pre_pts) + trans_loss
    loss = config["weight_loss"] * loss

    return loss


def sample_cube_points(n, size=100):
    coords = np.arange(-(n // 2), n // 2 + 1, dtype=np.float32) * (size / (n - 1))  # shape: (n,)
    assert coords[n // 2] == 0.0, "Center coordinate is not exactly 0!"
    grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
    points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return torch.from_numpy(points)


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

label_transformer_mix = LabelTransformMix2(global_config['norm_params'])
# volume_dir_2 = r"data/voxel_data/spine107_img.nii.gz"
volume_dir_2 = r"data/voxel_data/uniformed_liver_6.nii.gz"

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


# 联合模型
model = GridModel(model_config=global_config['model_config'], num_classes=6, in_channel=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=global_config["lr"])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=patience,  # ← 从 1 改为 200，给模型更多收敛时间
    min_lr=min_lr,  # ← 设置下限，防止过小
)
# ================= 【关键优化】预生成标准参考图像 =================

# PA 标准图 (零变换)
rot_pa_clean = torch.zeros(1, 3, device=device)  # [1, 3]
trans_pa_clean = torch.zeros(1, 3, device=device)  # [1, 3]
img_pa_clean = drr_pa(rot_pa_clean, trans_pa_clean, parameterization="euler_angles", convention="ZXY", degrees=True)
img_pa_clean = norm_img(img_pa_clean)  # [1, H, W] 或 [1, 1, H, W]

# RLAT 标准图 (零变换)
rot_rlat_clean = torch.zeros(1, 3, device=device)
trans_rlat_clean = torch.zeros(1, 3, device=device)
img_rlat_clean = drr_rlat(rot_rlat_clean, trans_rlat_clean, parameterization="euler_angles", convention="ZXY", degrees=True)
img_rlat_clean = norm_img(img_rlat_clean)  # [1, H, W] 或 [1, 1, H, W]

# 确保通道维度 [1, 1, H, W]
if img_pa_clean.dim() == 3:
    img_pa_clean = img_pa_clean.unsqueeze(1)
if img_rlat_clean.dim() == 3:
    img_rlat_clean = img_rlat_clean.unsqueeze(1)

# # 扩展到 batch_size 维度，方便后续拼接 [B, 1, H, W]
# img_pa_clean_batch = img_pa_clean.expand(batch_size, -1, -1, -1).contiguous()
# img_rlat_clean_batch = img_rlat_clean.expand(batch_size, -1, -1, -1).contiguous()

# ===  统一配置保存路径 ===
base_checkpoint_dir = "mix3_diff_uliver6_model"
os.makedirs(base_checkpoint_dir, exist_ok=True)

# 子目录：每个模型独立文件夹（便于管理）
checkpoint_dir_mix = os.path.join(base_checkpoint_dir, "mix_view")
os.makedirs(checkpoint_dir_mix, exist_ok=True)


# === 初始化训练状态 ===
max_steps = global_config['max_steps']
pts = sample_cube_points(3).to(device)

# 确保 batch_size 是偶数
assert batch_size % 2 == 0, "Batch size must be even to split 50/50"
half_batch = batch_size // 2


losses_mix = []
best_loss_mix = float('inf')


# 日志打印频率
log_interval = global_config.get('log_interval', 100)  # 默认每 100 步打印一次

for i in range(max_steps):

    # --- PA 视角部分 ---
    rot_noise_pa = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_pa = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)

    rotations_pa = rot_noise_pa * rota_range_t
    translations_pa = trans_noise_pa * trans_range_t

    rot_pa,trans_pa = get_transformer_noise(rotations_pa, translations_pa, R_ct2osz)
    label_rot_pa, label_trans_pa = norm_label(rot_pa, trans_pa)

    img_pa_noisy = drr_pa(
        rotations_pa,
        translations_pa,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img_pa_noisy = norm_img(img_pa_noisy)
    label_pa = torch.cat([label_rot_pa,label_trans_pa], dim=1)  # [8, 6]


    # --- RLAT 视角部分 ---
    rot_noise_rlat = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_rlat = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)

    rotations_rlat = rot_noise_rlat * rota_range_t
    translations_rlat = trans_noise_rlat * trans_range_t

    rot_rlat, trans_rlat = get_transformer_noise(rotations_rlat, translations_rlat, R_ct2osc)
    label_rot_rlat, label_trans_rlat = norm_label(rot_rlat, trans_rlat)

    img_rlat_noisy = drr_rlat(
        rotations_rlat,
        translations_rlat,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img_rlat_noisy = norm_img(img_rlat_noisy)
    label_rlat = torch.cat([label_rot_rlat, label_trans_rlat], dim=1)  # [8, 6]

    # 计算图像差值
    img_pa_diff = img_pa_noisy - img_pa_clean
    img_rlat_diff = img_rlat_noisy - img_rlat_clean

    # ================= 数据拼接与处理 =================

    # 1. 拼接 Batch 维度 (PA + RLAT)
    img_noisy_mix = torch.cat([img_pa_noisy, img_rlat_noisy], dim=0)  # [B, 1, H, W]

    # 2. 拼接标准图像 (从预生成的批次中取用)
    # 前 half_batch 用 PA 标准图，后 half_batch 用 RLAT 标准图
    img_diff_mix = torch.cat([img_pa_diff, img_rlat_diff], dim=0)  # [B, 1, H, W]

    # 3. 标签拼接
    label_pa = torch.cat([label_rot_pa, label_trans_pa], dim=1)
    label_rlat = torch.cat([label_rot_rlat, label_trans_rlat], dim=1)
    label_mix = torch.cat([label_pa, label_rlat], dim=0)  # [B, 6]

    # 4. 打乱顺序 (保持噪声图和标准图对应关系一致)
    shuffle_idx = torch.randperm(batch_size, device=device)
    img_noisy_mix = img_noisy_mix[shuffle_idx]
    img_diff_mix = img_diff_mix[shuffle_idx]  # 标准图也要同步打乱
    label_mix = label_mix[shuffle_idx]

    # 5. 通道拼接：形成双通道输入 [B, 2, H, W]
    input_dual_channel = torch.cat([img_noisy_mix, img_diff_mix], dim=1)


    # ================= 模型训练 =================
    model.train()
    output_mix = model(input_dual_channel)
    loss_mix = get_mse_loss(global_config, label_transformer_mix, label_mix, output_mix, pts)
    loss_mix_val = loss_mix.item()

    optimizer.zero_grad()
    loss_mix.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss_mix_val)
    losses_mix.append(loss_mix_val)


    # ================= 定期输出训练指标 =================
    if (i + 1) % log_interval == 0 or (i + 1) == max_steps:
        with torch.no_grad():
            # 混合模型监控
            out_range_mix = f"[{output_mix.min().item():.3f}, {output_mix.max().item():.3f}]"
            lbl_range_mix = f"[{label_mix.min().item():.3f}, {label_mix.max().item():.3f}]"
            grad_norm_mix = torch.norm(model.stem[0].weight.grad).item() if model.stem[
                                                                                0].weight.grad is not None else 0.0



        print(f"\n{'=' * 60}")
        print(f"Step [{i + 1:6d}/{max_steps}]")
        print(f"{'-' * 60}")
        print(f" MIX_VIEW | Loss: {loss_mix_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Out: {out_range_mix} | Grad: {grad_norm_mix:.4f}")
        print(f"{'=' * 60}\n")

    # ================= 保存最佳模型 =================
    # --- 混合模型 ---
    if loss_mix_val < best_loss_mix:
        improvement = (best_loss_mix - loss_mix_val) / best_loss_mix * 100 if best_loss_mix != float('inf') else 0
        best_loss_mix = loss_mix_val

        # 保存模型
        torch.save({
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss_mix,
            'config': global_config
        }, os.path.join(checkpoint_dir_mix, "best_model.pth"))

        # 🎉 输出最佳迭代信息
        with torch.no_grad():
            out_range = f"[{output_mix.min().item():.4f}, {output_mix.max().item():.4f}]"
            lbl_range = f"[{label_mix.min().item():.4f}, {label_mix.max().item():.4f}]"
            grad_norm = torch.norm(model.stem[0].weight.grad).item() if model.stem[0].weight.grad is not None else 0.0

        print(f"[MIX_VIEW]  NEW BEST at Step {i + 1}!")
        print(f" Loss: {best_loss_mix:.8f} (↓{improvement:.2f}% vs prev best)")
        print(f" Output: {out_range} | Label: {lbl_range} | GradNorm: {grad_norm:.4f}")





    # ================= 定期保存中间检查点 =================
    if (i + 1) % global_config.get('save_interval', 1000) == 0 and (i + 1) < max_steps:
        # 混合模型
        torch.save(model.state_dict(), os.path.join(checkpoint_dir_mix, f"model_step_{i + 1}.pth"))
        print(f"💾 Checkpoint saved at step {i + 1}")

# === 4. 训练结束：保存最终模型和日志 ===
print("\n🏁 Training finished. Saving final models and logs...")

# --- 保存最终模型权重 ---
torch.save(model.state_dict(), os.path.join(checkpoint_dir_mix, "final_model.pth"))

# --- 保存 losses 日志 (JSON + TXT 双格式，便于绘图和查看) ---

for name, losses, dir_path in [
    ("mix_view", losses_mix, checkpoint_dir_mix)
]:
    # JSON 格式（便于 Python 读取绘图）
    with open(os.path.join(dir_path, "losses.json"), "w") as f:
        json.dump({"steps": list(range(1, len(losses) + 1)), "losses": losses}, f, indent=2)

    # TXT 格式（便于快速查看）
    with open(os.path.join(dir_path, "losses.txt"), "w") as f:
        f.write("step,loss\n")
        for step, loss_val in enumerate(losses, start=1):
            f.write(f"{step},{loss_val:.8f}\n")

    print(f"📊 {name} logs saved to {dir_path}")

# --- 保存训练汇总信息 ---
summary = {
    "total_steps": max_steps,
    "batch_size": batch_size,
    "half_batch": half_batch,
    "best_losses": {
        "mix_view": best_loss_mix
    },
    "final_losses": {
        "mix_view": losses_mix[-1] if losses_mix else None
    }
}
with open(os.path.join(base_checkpoint_dir, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ All models and logs saved to: {base_checkpoint_dir}")
print(f"📁 Directory structure:")
print(f"   {base_checkpoint_dir}/")
print(f"   ├── mix_view/      → best_model.pth, final_model.pth, losses.json, losses.txt")
print(f"   └── training_summary.json")