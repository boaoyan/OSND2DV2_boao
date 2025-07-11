import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QMainWindow
from PyQt5.QtGui import QPixmap, QImage

from view2D.utils import convert_gray_to_qimage
from view2D.utils.img_processing import add_circular_mask, normalize_to_255_gray_single


def convert_cv_qimage(cv_img):
    """将 OpenCV 图像转换为 QImage"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Image in QGraphicsScene")
        self.setGeometry(100, 100, 800, 600)

        # 创建 QGraphicsScene 和 QGraphicsView
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

        # 读取图像
        self.cv_image = cv2.imread("data/ct_img/x270_y90_z270.png", 0)  # 替换为你自己的图片路径
        self.cv_image = add_circular_mask(255-self.cv_image)
        if self.cv_image is None:
            print("无法加载图像！")
            return

        # 转换图像
        qimage = convert_gray_to_qimage(self.cv_image)
        pixmap = QPixmap.fromImage(qimage)

        # 添加图像到场景中
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())