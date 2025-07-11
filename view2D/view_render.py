import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

from view2D.utils import convert_gray_to_qimage
from view2D.utils.img_processing import apply_circular_mask, darken_image


class ViewRender:
    def __init__(self, qt_view: QGraphicsView, origin_img):
        self.qt_view = qt_view
        self.qt_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.qt_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # section 1 在控件中添加场景
        self.qt_scene = QGraphicsScene()
        self.qt_view.setScene(self.qt_scene)
        self.qt_scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        # section 2.1 添加背景投影图
        origin_img = apply_circular_mask(darken_image(255 - origin_img))
        self.origin_img = origin_img
        self.bg_im = None
        self.update_im(self.origin_img)

    def update_im(self, im, uv=np.array([0, 0]), is_set_scene_size=False):
        """
            更新背景投影图
        :param im: 在视口中应该添加的图像
        :param uv: 图像的左上角在场景中的位置
        :param is_set_scene_size: 是否需要根据图片的大小设置场景的大小
        :return:
        """
        if self.bg_im is not None:
            self.qt_scene.removeItem(self.bg_im)
        # 转换图像
        qimage = convert_gray_to_qimage(im)
        pixmap = QPixmap.fromImage(qimage)
        # 添加图像到场景中
        self.bg_im = QGraphicsPixmapItem(pixmap)
        self.bg_im.setPos(uv[0], uv[1])
        self.qt_scene.addItem(self.bg_im)
        if is_set_scene_size:
            # 根据图片的大小设置场景的大小
            u, v = im.shape
            # print("图片的大小为：", u, v)
            self.qt_scene.setSceneRect(QRectF(uv[0], uv[1], v, u))
        self.bg_im.setZValue(-1)

    def resize_event(self):
        if self.bg_im is None:
            return
        self.qt_view.fitInView(self.bg_im, Qt.AspectRatioMode.KeepAspectRatio)