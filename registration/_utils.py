"""
记录需要用到的常用工具函数
1 差异图生成
2 保存 tensor 为图像
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        # squeeze extra dimensions: keep only spatial (H, W) or (H, W, C)
        if x.ndim == 4:  # [B, C, H, W]
            x = x.squeeze(0).squeeze(0)  # → [H, W]
        elif x.ndim == 3:  # [C, H, W] or [H, W, C]
            if x.shape[0] == 1:  # [1, H, W]
                x = x.squeeze(0)
            elif x.shape[2] == 1:  # [H, W, 1]
                x = x.squeeze(-1)
        elif x.ndim == 2:
            pass  # already [H, W]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        raise TypeError(f"Input must be np.ndarray or torch.Tensor, got {type(x)}")
    return x

# ========== 差异图函数==========
def create_difference_map(img_estimated, img_true, show=False):
    img_estimated = to_numpy(img_estimated)
    img_true = to_numpy(img_true)
    diff = img_estimated.astype(np.float32) - img_true.astype(np.float32)
    red = np.clip(diff, 0, 255).astype(np.uint8)
    blue = np.clip(-diff, 0, 255).astype(np.uint8)

    if red.max() > 0:
        red_norm = (red / red.max()) * 255
    else:
        red_norm = red
    if blue.max() > 0:
        blue_norm = (blue / blue.max()) * 255
    else:
        blue_norm = blue

    red_norm = red_norm.astype(np.uint8)
    blue_norm = blue_norm.astype(np.uint8)

    diff_bgr = np.stack([blue_norm, np.zeros_like(red_norm), red_norm], axis=-1)

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(red_norm, cmap='Reds')
        axes[0].set_title('Positive Difference (Red)')
        axes[0].axis('off')

        axes[1].imshow(blue_norm, cmap='Blues')
        axes[1].set_title('Negative Difference (Blue)')
        axes[1].axis('off')

        final_rgb = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2RGB)
        axes[2].imshow(final_rgb)
        axes[2].set_title('Difference Map')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

    return diff_bgr


# ========== 保存 tensor 为图像 ==========
def save_tensor_as_image_pil(tensor, filepath, auto_scale=True):
    img = tensor.squeeze().detach().cpu().numpy()
    if auto_scale:
        if img.dtype in (np.float32, np.float64):
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    image = Image.fromarray(img)
    image.save(filepath)