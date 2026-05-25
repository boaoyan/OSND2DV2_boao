import numpy as np
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from grid_efficientnet.grid_efficientb0_model import GridModel
from label_transform_mix import LabelTransformMix
from projector.drr import DRR
from projector.read_data import read
from projector.pose import convert
from registration.label_transform_mix_2 import LabelTransformMix2


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
        "model_config": {
            'edffn': 1,
            'eca': 0,
            'conv_mlp': 0
        },
        "noise_params": {
            'trans_noise_range': [25, 25, 0],
            'rota_noise_range': [0, 0, 0]
        },
        "norm_params": {
            'trans_noise_norm': [25, 25, 25],
            'rota_noise_norm': [1, 1, 1]
        }
    }


# def get_mse_loss(config, label_transformer_mix, labels, outputs, pts):
#     # 1️⃣ 获取 pts 的设备（作为基准设备）
#     device = outputs.device  # ← 以模型输出为基准（通常在 GPU）
#     pts = pts.to(device)  # ← 确保 pts 在同一设备
#
#     pre_rota, pre_trans = label_transformer_mix.label2real(outputs)
#     tru_rota, tru_trans = label_transformer_mix.label2real(labels)
#
#     # 2️⃣ 转换得到 pose 对象
#     tru_pose = convert(tru_rota, tru_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
#     pre_pose = convert(pre_rota, pre_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
#
#     # 3️⃣ 🔥 关键修复：将 rotation 和 translation 迁移到 pts 同一设备
#     tru_rotation = tru_pose.rotation.mT.to(device)
#     pre_rotation = pre_pose.rotation.mT.to(device)
#     tru_translation = tru_pose.translation.to(device)
#     pre_translation = pre_pose.translation.to(device)
#
#     # 4️⃣ 执行矩阵运算（现在所有张量都在同一设备）
#     tru_pts = torch.matmul(tru_rotation, pts.T)
#     pre_pts = torch.matmul(pre_rotation, pts.T)
#
#     # 5️⃣ 计算损失
#     trans_loss = F.mse_loss(tru_translation, pre_translation)
#     loss = F.mse_loss(tru_pts, pre_pts) + trans_loss
#     loss = config["weight_loss"] * loss
#
#     return loss
def get_mse_loss_2dof(config, label_transformer, labels, outputs, pts,
                      valid_trans_dims=[0, 1], valid_rot_dims=None):
    """
    valid_trans_dims: 有效的平移维度 [0]=X, [1]=Y, [2]=Z
    valid_rot_dims: 有效的旋转维度，None表示完全忽略旋转loss
    """
    device = outputs.device
    pts = pts.to(device)

    pre_rota, pre_trans = label_transformer.label2real(outputs)
    tru_rota, tru_trans = label_transformer.label2real(labels)

    # ✅ 平移loss：只计算有效维度
    trans_loss = F.mse_loss(
        tru_trans[:, valid_trans_dims],
        pre_trans[:, valid_trans_dims]
    )

    # ✅ 旋转loss：可选弱化或完全忽略
    if valid_rot_dims is not None and len(valid_rot_dims) > 0:
        rot_loss = F.mse_loss(
            tru_rota[:, valid_rot_dims],
            pre_rota[:, valid_rot_dims]
        ) * 1e-3  # 弱化权重
    else:
        rot_loss = torch.tensor(0.0, device=device)  # 完全忽略

    # ✅ 点变换loss：使用零旋转 + 有效平移计算
    zero_rot = torch.zeros_like(tru_rota)
    ref_pose = convert(zero_rot, tru_trans, parameterization="euler_angles",
                       convention="ZXY", degrees=True)
    pred_pose = convert(zero_rot, pre_trans, parameterization="euler_angles",
                        convention="ZXY", degrees=True)

    ref_pts = torch.matmul(ref_pose.rotation.mT.to(device), pts.T)
    pred_pts = torch.matmul(pred_pose.rotation.mT.to(device), pts.T)
    point_loss = F.mse_loss(ref_pts, pred_pts)

    total_loss = config["weight_loss"] * (point_loss + trans_loss + rot_loss)
    return total_loss


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

patience = global_config['patience']
min_lr = global_config['min_lr']
height = global_config['height']
delx = global_config['delx']
batch_size = global_config['batch_size']

label_transformer_mix = LabelTransformMix2(global_config['norm_params'])
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
model = GridModel(model_config=global_config['model_config'], num_classes=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=global_config["lr"])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=patience,  # ← 从 1 改为 200，给模型更多收敛时间
    min_lr=min_lr,  # ← 设置下限，防止过小
)

# ===  统一配置保存路径 ===
base_checkpoint_dir = "mix1_2dof_uliver6_model"
os.makedirs(base_checkpoint_dir, exist_ok=True)

# 子目录：每个模型独立文件夹（便于管理）
checkpoint_dir_mix = os.path.join(base_checkpoint_dir, "mix1_view")
os.makedirs(checkpoint_dir_mix, exist_ok=True)

# === 初始化训练状态 ===
max_steps = global_config['max_steps']
pts = sample_cube_points(3).to(device)

# 确保 batch_size 是偶数
assert batch_size % 2 == 0, "Batch size must be even to split 50/50"
half_batch = batch_size // 2

# 三个模型的 losses 记录列表
losses_mix = []

# 三个模型的最佳 loss 记录
best_loss_mix = float('inf')

# 日志打印频率
log_interval = global_config.get('log_interval', 100)  # 默认每 100 步打印一次

for i in range(max_steps):

    # --- PA 视角部分 ---
    rot_noise_pa = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_pa = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)

    img_pa = drr_pa(
        rot_noise_pa * rota_range_t,
        trans_noise_pa * trans_range_t,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img_pa = norm_img(img_pa)
    label_pa = torch.cat([rot_noise_pa, trans_noise_pa], dim=1)  # [8, 6]

    # --- RLAT 视角部分 ---
    rot_noise_rlat = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)
    trans_noise_rlat = torch.empty(half_batch, 3, device=device).uniform_(-1.0, 1.0)

    img_rlat = drr_rlat(
        rot_noise_rlat * rota_range_t,
        trans_noise_rlat * trans_range_t,
        parameterization="euler_angles", convention="ZXY", degrees=True
    )
    img_rlat = norm_img(img_rlat)
    label_rlat = torch.cat([rot_noise_rlat, trans_noise_rlat], dim=1)  # [8, 6]

    # --- 混合模型数据：拼接 + 打乱 ---
    img_mix = torch.cat([img_pa, img_rlat], dim=0)  # [16, H, W]
    label_mix = torch.cat([label_pa, label_rlat], dim=0)  # [16, 6]
    shuffle_idx = torch.randperm(batch_size, device=device)
    img_mix = img_mix[shuffle_idx]
    label_mix = label_mix[shuffle_idx]

    # ================= 模型 1: 混合视角训练 =================
    model.train()
    output_mix = model(img_mix)
    loss_mix = get_mse_loss_2dof(global_config, label_transformer_mix, label_mix, output_mix, pts)
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

        print(f" [MIX_VIEW]  NEW BEST at Step {i + 1}!")
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