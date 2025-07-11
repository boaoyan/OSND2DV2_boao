"""
负责图像显示过程中的格式转换问题
1 ndarray转为pixmap
"""
import cv2
from PyQt5.QtGui import QImage


def convert_cv_qimage(cv_img):
    """将 OpenCV 图像转换为 QImage"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


def convert_gray_to_qimage(cv_img):
    """
    将 OpenCV 单通道灰度图像转换为 QImage（RGB 格式）
    """
    # 将单通道图像转换为三通道
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

    h, w = cv_img.shape
    ch = 3
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
