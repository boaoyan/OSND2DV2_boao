import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path


# ================= 1. 数据集类（含 Log1p 归一化） =================
class PoseMSDDataset(Dataset):
    def __init__(self, csv_path, train_ratio=0.8, random_seed=42):
        df = pd.read_csv(csv_path)
        feature_cols = ['t1_x', 't1_y', 't1_z', 'r1_z', 'r1_x', 'r1_y',
                        't2_x', 't2_y', 't2_z', 'r2_z', 'r2_x', 'r2_y']
        self.X_raw = df[feature_cols].values.astype(np.float32)
        self.y_raw = df['msd_pose'].values.astype(np.float64)

        # 输入 Z-Score
        self.x_mean = self.X_raw.mean(axis=0)
        self.x_std = self.X_raw.std(axis=0) + 1e-8
        X_norm = (self.X_raw - self.x_mean) / self.x_std

        # 输出 Log1p 变换（核心：解决右偏分布）
        self.y_log = np.log1p(self.y_raw)
        self.y_mean = float(self.y_log.mean())
        self.y_std = float(self.y_log.std()) + 1e-8
        y_norm = (self.y_log - self.y_mean) / self.y_std

        # 划分
        np.random.seed(random_seed)
        idx = np.random.permutation(len(X_norm))
        split = int(train_ratio * len(X_norm))
        train_idx, val_idx = idx[:split], idx[split:]

        self.X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
        self.y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32).unsqueeze(1)
        self.X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32)
        self.y_val = torch.tensor(y_norm[val_idx], dtype=torch.float32).unsqueeze(1)

        print(f"✅ 加载完成 | 训练: {len(self.X_train)} | 验证: {len(self.X_val)}")
        print(f"📊 MSD 原始范围: [{self.y_raw.min():.2f}, {self.y_raw.max():.2f}] mm²")

    def __len__(self): return len(self.X_train)

    def __getitem__(self, idx): return self.X_train[idx], self.y_train[idx]

    def get_val_loader(self, batch_size=2048):
        return DataLoader(TensorDataset(self.X_val, self.y_val),
                          batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def denormalize_msd(self, y_norm):
        """将归一化对数输出还原为原始 mm²"""
        if isinstance(y_norm, torch.Tensor): y_norm = y_norm.detach().cpu().numpy()
        y_log = y_norm * self.y_std + self.y_mean
        return np.expm1(y_log)


# ================= 2. 模型定义 =================
class PoseMSDNet(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x): return self.net(x)


# ================= 3. 训练主流程 =================
def train_pose_msd(csv_path, save_dir, epochs=80, batch_size=2048, lr=1e-3, patience=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 设备: {device} | Batch: {batch_size} | LR: {lr}")

    # 数据
    dataset = PoseMSDDataset(csv_path)
    train_loader = DataLoader(TensorDataset(dataset.X_train, dataset.y_train),
                              batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = dataset.get_val_loader(batch_size=batch_size)

    # 模型 & 优化器
    model = PoseMSDNet().to(device)
    if hasattr(torch, 'compile'): model = torch.compile(model)  # PyTorch 2.0+ 加速
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2, factor=0.5, verbose=True)
    scaler = GradScaler()  # AMP 混合精度

    # 训练循环
    best_val_loss = float('inf')
    no_improve = 0
    tr_hist, vl_hist = [], []
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"⏳ 开始训练...")
    t_start = time.time()
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_tr_loss += loss.item() * xb.size(0)
        tr_hist.append(epoch_tr_loss / len(train_loader.dataset))

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
        vl_hist.append(val_loss / len(val_loader.dataset))
        scheduler.step(vl_hist[-1])

        # Early Stopping & Save Best
        if vl_hist[-1] < best_val_loss:
            best_val_loss = vl_hist[-1]
            torch.save({
                'model_state_dict': model.state_dict(),
                'dataset_stats': {'x_mean': dataset.x_mean.tolist(), 'x_std': dataset.x_std.tolist(),
                                  'y_mean': dataset.y_mean, 'y_std': dataset.y_std},
                'epochs': epoch + 1, 'val_loss': best_val_loss
            }, os.path.join(save_dir, 'best_pose_msd.pth'))
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d} | Train: {tr_hist[-1]:.4e} | Val: {vl_hist[-1]:.4e} | LR: {optimizer.param_groups[0]['lr']:.1e}")
        if no_improve >= patience:
            print(f"⏹️ 早停触发 @ Epoch {epoch + 1}")
            break

    print(f"✅ 训练完成 | 总耗时: {time.time() - t_start:.1f}s | 最佳 Val Loss: {best_val_loss:.4e}")
    return model, tr_hist, vl_hist, dataset


# ================= 4. 评估与可视化 =================
def evaluate_and_plot(model, dataset, save_dir):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        y_pred_norm = model(dataset.X_val.to(device))
        y_pred = dataset.denormalize_msd(y_pred_norm)
        y_true = dataset.y_raw[int(0.8 * len(dataset.y_raw)):]  # 对齐验证集索引

    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rel_err = np.abs(y_true - y_pred) / (y_true + 1e-8) * 100

    print(f"\n📊 验证集评估 (原始 mm²):")
    print(f"   • R²: {r2:.4f} | MAE: {mae:.2f} mm²")
    print(f"   • 平均相对误差: {rel_err.mean():.2f}% | 最大: {rel_err.max():.2f}%")

    # 散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, c=rel_err, cmap='viridis_r', s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect')
    plt.xlabel('True MSD (mm²)');
    plt.ylabel('Predicted MSD (mm²)')
    plt.colorbar(label='Relative Error (%)')
    plt.title(f'Pose MSD Prediction (R²={r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pose_msd_scatter.png'), dpi=150)
    plt.close()

    # 损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(tr_hist, label='Train');
    plt.plot(vl_hist, label='Val')
    plt.yscale('log');
    plt.xlabel('Epoch');
    plt.ylabel('MSE Loss (Log-Space)')
    plt.legend();
    plt.grid(True, alpha=0.3);
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pose_msd_loss.png'), dpi=150)
    plt.close()


# ================= 5. 执行入口 =================
if __name__ == "__main__":
    CSV_PATH = "../data/loss_data/pose_msd_dataset.csv"
    SAVE_DIR = "../data/loss_data/models/pose_msd_v1/"

    model, tr_hist, vl_hist, dataset = train_pose_msd(
        csv_path=CSV_PATH, save_dir=SAVE_DIR,
        epochs=80, batch_size=2048, lr=1e-3, patience=12
    )
    evaluate_and_plot(model, dataset, SAVE_DIR)