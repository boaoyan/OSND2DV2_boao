
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from registration.mlp.loss_mlp import CompactPoseLossMLP

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings


def train_compact_pose_loss_mlp(
        dataset_path: str,
        model_save_path: str = "compact_pose_loss_mlp.pth",
        val_ratio: float = 0.1,
        target_log_offset: float = 1e-8,
        epochs: int = 100,
        batch_size: int = 4096,
        lr: float = 3e-3,  # 🔑 维度降低后可适当提高学习率
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        early_stop_patience: int = 15,
        early_stop_min_delta: float = 1e-4,
        device: str = "cuda",
        seed: int = 42
):
    torch.manual_seed(seed);
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练设备: {device} | 输入维度: 12 (直接拼接)")

    # 1️⃣ 加载数据
    data = torch.load(dataset_path, map_location='cpu')
    gt_all = data["gt"].float()
    pred_all = data["pred"].float()
    loss_all = data["loss"].float()

    # 2️⃣ 目标值对数变换
    target_log = torch.log(loss_all + target_log_offset)

    # 3️⃣ 划分数据集
    n = len(loss_all)
    idx = torch.randperm(n)
    val_sz = int(n * val_ratio)
    train_ds = TensorDataset(gt_all[idx[val_sz:]], pred_all[idx[val_sz:]], target_log[idx[val_sz:]])
    val_ds = TensorDataset(gt_all[idx[:val_sz]], pred_all[idx[:val_sz]], target_log[idx[:val_sz]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    print(f"📊 数据划分: 训练 {len(train_ds):,} / 验证 {len(val_ds):,}")

    # 4️⃣ 初始化模型
    model = CompactPoseLossMLP(
        hidden_dims=(256, 128, 64),
        dropout=0.05,
        output_bias_init=-7.0
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    criterion = nn.MSELoss()

    # 5️⃣ 训练循环
    history = {"train": [], "val": [], "lr": []}
    best_val, patience, best_state = float('inf'), 0, None
    pbar = tqdm(range(epochs), desc="训练进度")

    for epoch in pbar:
        model.train();
        train_losses = []
        for gt, pred, tgt in train_loader:
            gt, pred, tgt = gt.to(device), pred.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)

            pred_log = model(gt, pred).squeeze(-1)
            loss = criterion(pred_log, tgt)
            loss.backward()
            if grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval();
        val_losses = []
        with torch.no_grad():
            for gt, pred, tgt in val_loader:
                gt, pred, tgt = gt.to(device), pred.to(device), tgt.to(device)
                val_losses.append(criterion(model(gt, pred).squeeze(-1), tgt).item())

        avg_t, avg_v = np.mean(train_losses), np.mean(val_losses)
        history["train"].append(avg_t);
        history["val"].append(avg_v)
        history["lr"].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 10 == 0:
            pbar.set_postfix(
                {"train": f"{avg_t:.4f}", "val": f"{avg_v:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if avg_v < best_val - early_stop_min_delta:
            best_val, patience, best_state = avg_v, 0, model.state_dict().copy()
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"\n⏹️ 早停触发 @ epoch {epoch + 1}");
                break

    if best_state: model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "history": history, "best_val": best_val}, model_save_path)
    print(f"\n✅ 训练完成! 最佳验证 Loss: {best_val:.4f} (log空间)")
    return model, history


def inspect_normalized_dataset(dataset_path: str):
    """快速检查归一化数据集的统计特性"""
    data = torch.load(dataset_path, map_location='cpu')
    gt, pred, loss = data["gt"], data["pred"], data["loss"]

    print(f"\n🔍 数据集质检: {dataset_path}")
    print("=" * 60)
    print(f"📦 样本数: {len(loss):,}")

    # 位姿误差统计
    delta = (pred - gt).abs()
    print(f"\n🔄 旋转误差 (°):")
    print(f"   Z: mean={delta[:, 0].mean():.3f}, max={delta[:, 0].max():.3f}")
    print(f"   X: mean={delta[:, 1].mean():.3f}, max={delta[:, 1].max():.3f}")
    print(f"   Y: mean={delta[:, 2].mean():.3f}, max={delta[:, 2].max():.3f}")

    print(f"\n📍 平移误差 (归一化单位):")
    print(f"   X: mean={delta[:, 3].mean():.4f}, max={delta[:, 3].max():.4f}")
    print(f"   Y: mean={delta[:, 4].mean():.4f}, max={delta[:, 4].max():.4f}")
    print(f"   Z: mean={delta[:, 5].mean():.4f}, max={delta[:, 5].max():.4f}")

    # Loss 分布（关键！）
    print(f"\n📊 Loss 真值分布 (归一化空间):")
    print(f"   min: {loss.min():.2e}")
    print(f"   10%: {loss.quantile(0.1):.2e}")
    print(f"   50%: {loss.median():.2e}")
    print(f"   90%: {loss.quantile(0.9):.2e}")
    print(f"   max: {loss.max():.2e}")

    # 检查 NaN/Inf
    assert not torch.isnan(loss).any(), "❌ 数据集中存在 NaN Loss!"
    assert not torch.isinf(loss).any(), "❌ 数据集中存在 Inf Loss!"
    print("\n✅ 数据格式完整，无异常值")


def load_and_predict(model_path: str, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    加载训练好的模型并预测归一化 Loss
    :param gt: [B, 6] 归一化标准噪声
    :param pred: [B, 6] 归一化预测噪声
    :return: [B] 预测的归一化 Loss
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint["config"]

    model = CompactPoseLossMLP(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_log = model(gt, pred).squeeze(-1)  # [B]
        # 🔑 若需还原物理空间 Loss（可选）: loss_phys = torch.exp(pred_log) - offset
        return pred_log  # 保持对数空间输出更稳定

if __name__ == "__main__":
    # 0. 数据质检（必做！）
    inspect_normalized_dataset("../data/noise_norm.csv")

    # # 1. 训练模型
    # model, history = train_compact_pose_loss_mlp(
    #     dataset_path="./data/pose_loss_liver_norm_30k.pt",
    #     model_save_path="./models/pose_loss_mlp_liver_norm.pth",
    #     epochs=100,
    #     batch_size=4096,
    #     lr=2e-3,
    #     early_stop_patience=15,
    #     device="cuda"
    # )

    # # 2. 可视化训练曲线（可选）
    #
    # epochs_plot = range(1, len(history["train_loss"]) + 1)
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_plot, history["train_loss"], label="Train")
    # plt.plot(epochs_plot, history["val_loss"], label="Val")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss (log space)")
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_plot, history["lr"])
    # plt.xlabel("Epoch")
    # plt.ylabel("Learning Rate")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("training_curves.png", dpi=150)
    # print("📈 训练曲线已保存: training_curves.png")

    # # 3. 推理测试
    # test_gt = torch.randn(10, 6) * torch.tensor([2, 2, 2, 0.01, 0.01, 0.01])  # 小误差样本
    # test_pred = test_gt + torch.randn(10, 6) * torch.tensor([0.5, 0.5, 0.5, 0.002, 0.002, 0.002])
    #
    # pred_loss_log = load_and_predict("./models/pose_loss_mlp_liver_norm.pth", test_gt, test_pred)
    # pred_loss = torch.exp(pred_loss_log)  # 还原为线性空间（可选）
    #
    # print(f"\n🔮 推理示例:")
    # print(f"预测 Loss (log): {pred_loss_log.mean():.4f} ± {pred_loss_log.std():.4f}")
    # print(f"预测 Loss (linear): {pred_loss.mean():.2e} ± {pred_loss.std():.2e}")