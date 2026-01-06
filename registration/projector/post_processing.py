import torch
import torch.nn.functional as F

def apply_circular_mask(img_tensor, radius_ratio=0.99, softness=None):
    """
    在输入的图像 tensor 上应用一个圆形遮罩，支持软边缘过渡。

    参数:
        img_tensor (torch.Tensor): 输入图像张量，形状为 (H, W)、(C, H, W) 或 (B, C, H, W)
        radius_ratio (float): 圆形遮罩半径占图像短边的比例（默认 0.95）
        softness (float): 边缘模糊程度，值越大边缘越柔和（默认 1.0）

    返回:
        masked_img (torch.Tensor): 应用遮罩后的图像，保持原始形状
    """
    assert isinstance(img_tensor, torch.Tensor), "Input must be a torch.Tensor"
    device = img_tensor.device
    dims = img_tensor.dim()

    # 自动识别输入维度并提取 B, C, H, W
    if dims == 2:
        H, W = img_tensor.shape
        B = 1
        C = 1
        img = img_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    elif dims == 3:
        C, H, W = img_tensor.shape
        B = 1
        img = img_tensor.unsqueeze(0)  # -> (1, C, H, W)
    elif dims == 4:
        B, C, H, W = img_tensor.shape
        img = img_tensor
    else:
        raise ValueError("Input tensor must be 2D, 3D or 4D")

    # 创建坐标网格
    Y, X = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    center = (H // 2, W // 2)
    max_radius = min(H, W) // 2 * radius_ratio

    # 计算每个像素到中心的距离平方
    dist_from_center = (X - center[1])**2 + (Y - center[0])**2
    mask = dist_from_center <= max_radius**2

    # 转换为浮点型，并扩展到 (B, C, H, W)
    mask = mask.float().to(device)
    mask = mask.expand(B, C, H, W)

    if softness is not None:
        # 添加高斯模糊来生成软边缘
        kernel_size = int(11 * softness)  # 模糊核大小
        if kernel_size % 2 == 0:
            kernel_size += 1  # 必须是奇数
        sigma = 3.0 * softness

        # 使用 PyTorch 的卷积实现高斯模糊
        padding = kernel_size // 2
        kernel = _get_gaussian_kernel2d(kernel_size, sigma, device=device)

        # 对每个 batch 和通道单独做卷积
        mask = mask.view(B * C, 1, H, W)
        soft_mask = F.conv2d(mask, kernel, padding=padding, groups=1)
        soft_mask = soft_mask.view(B, C, H, W).clamp(0, 1)

        # 应用软遮罩
        masked_img = img * soft_mask
    else:
        masked_img = img * mask

    # 恢复原始输入形状
    if dims == 2:
        return masked_img.squeeze(0).squeeze(0)
    elif dims == 3:
        return masked_img.squeeze(0)
    else:
        return masked_img


def _get_gaussian_kernel1d(kernel_size, sigma, device='cpu'):
    """生成一维高斯核"""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _get_gaussian_kernel2d(kernel_size, sigma, device='cpu'):
    """生成二维高斯核"""
    gk1d = _get_gaussian_kernel1d(kernel_size, sigma, device)
    gk2d = torch.outer(gk1d, gk1d)
    gk2d = gk2d.expand(1, 1, kernel_size, kernel_size)
    return gk2d

def normalize_to_255(img_tensor):
    """
    对 shape=[B, C, H, W] 的 tensor 图像，在 HxW 维度上归一化到 [0, 255]

    参数:
        img_tensor (torch.Tensor): 输入图像 tensor

    返回:
        torch.Tensor: 归一化后的 uint8 tensor
    """
    B, C, H, W = img_tensor.shape

    # 展开后两个维度为 HW
    x = img_tensor.view(B * C, -1)

    # 在最后两个维度上做归一化
    min_vals, _ = x.min(dim=1, keepdim=True)
    max_vals, _ = x.max(dim=1, keepdim=True)
    x = (x - min_vals) / (max_vals - min_vals + 1e-8)  # 防止除以零

    # 映射到 [0, 255] 并转为 uint8
    x = (x * 255).clamp(0, 255).to(torch.uint8)

    # 黑白反相
    # x = 255 - x

    # 恢复原始形状
    x = x.view(B, C, H, W)

    return x