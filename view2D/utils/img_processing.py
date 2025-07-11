import cv2
import numpy as np


def apply_circular_mask(img, center=None, radius=None, invert=False):
    # 创建全白遮罩(与图像同大小)
    mask = np.zeros_like(img)

    # 设置圆心和半径(如果未提供)
    if center is None:
        h, w = img.shape
        center = (w // 2, h // 2)
    if radius is None:
        h, w = img.shape
        radius = min(h, w) // 2

    # 绘制黑色圆形到遮罩上
    cv2.circle(mask, center, radius, 255, -1)

    # 如果需要反转遮罩
    if invert:
        mask = 255 - mask

    # 应用遮罩：保留圆内区域，圆外设为0(黑色)
    result = cv2.bitwise_and(img, mask)

    return result


def darken_image(img, factor=0.7):
    # 将图像转换为浮点型以便乘法运算
    img_float = img.astype(np.float32)
    # 调整亮度
    darkened = img_float * factor
    # 转换回8位无符号整型
    darkened = np.clip(darkened, 0, 255).astype(np.uint8)
    return darkened
