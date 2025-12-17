import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QColor, QPixmap
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLabel, QSizePolicy

from view2D.base_qt_item import BasePointItem, PJLineItem, SingleLineItem
from view2D.utils import convert_gray_to_qimage
from view2D.utils.CommunicatedSignal import Communicate
from view2D.utils.img_processing import apply_circular_mask, darken_image


# 已知视图中一条直线，求解其与该视图四条边界的交点
def get_line_in_img(m, c, img_size=512):
    # 计算射线和x=0, x=511, y = 0, y = 511的交点
    max_len = img_size - 1
    # 初始化矩阵存储四个交点
    cross_point = np.array([[0, c],
                            [max_len, c + m * max_len],
                            [-c / (m + 0.0000000001), 0],
                            [(max_len - c) / (m + 0.0000000001), max_len]])
    valid_point = []
    # 每个点取整之后的坐标需要在0到512之间，在才算有效点
    for i in range(4):
        if 0 <= int(cross_point[i][0]) <= max_len and 0 <= int(cross_point[i][1]) <= max_len:
            valid_point.append(cross_point[i])
    # print(np.array(valid_point).astype(np.uint64))
    # img = Image.fromarray(line_matrix)
    # img.show()
    return valid_point


class ViewRender:
    def __init__(self, qt_view: QGraphicsView, origin_img, uv_label: QLabel, rt_ct2o: str):
        self.qt_view = qt_view
        self.qt_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.qt_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.qt_view.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.qt_view.setFixedSize(400, 400)  # 强烈建议同时设置固定大小
        # section 1 在控件中添加场景
        self.qt_scene = QGraphicsScene()
        self.qt_view.setScene(self.qt_scene)
        self.qt_scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        # section 2 添加背景投影图
        origin_img = apply_circular_mask(darken_image(255 - origin_img))
        self.origin_img = origin_img
        self.bg_im = None
        self.update_im(self.origin_img)
        # section 3 规划功能
        self.signal = Communicate()
        self.qt_view.mousePressEvent = self.mouse_press_event
        # 处理选点功能
        self._scene_pos = None
        self._real_uv = None
        self.uv_label = uv_label
        self.guide_point_item = BasePointItem(self.qt_scene, "target")
        self.guide_point_item.set_item_visibility(False)
        # 键盘移动点位置
        self.qt_view.keyPressEvent = self.key_press_event
        # CT坐标系到光源坐标系的转换关系
        self.rt_ct2o = np.load(rt_ct2o)
        print("加载的转换矩阵为：", self.rt_ct2o)
        # 绘制投影线
        self.pj_line_item = PJLineItem(self.qt_scene)
        self.slope, self.intercept = None, None
        # 当前状态
        # none: 没有点；point: 有点，没有直线；line: 有点，有直线；both: 有点和投影线
        self.current_status = "none"
        # section 4 实时显示探针位置
        self.real_pin_item = BasePointItem(self.qt_scene, "pin")
        self.real_pin_item.set_item_visibility(False)
        self.real_dire_item = BasePointItem(self.qt_scene, "dire")
        self.real_dire_item.set_item_visibility(False)
        self.real_line_item = SingleLineItem(self.qt_scene)

    @property
    def scene_pos(self):
        return self._scene_pos

    @scene_pos.setter
    def scene_pos(self, value):
        self._scene_pos = value

    @property
    def real_uv(self):
        return self._real_uv

    @real_uv.setter
    def real_uv(self, value):
        self._real_uv = value

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

    def mouse_press_event(self, event):
        scene_pos = self.qt_view.mapToScene(event.pos())
        u = scene_pos.x()
        v = scene_pos.y()
        self.scene_pos = np.array([u, v])
        self.signal.widget_clicked.emit(self)

    def key_press_event(self, event):
        if event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_A:
            self.scene_pos[0] -= 1
        elif event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_D:
            self.scene_pos[0] += 1
        elif event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_W:
            self.scene_pos[1] -= 1
        elif event.key() == Qt.Key.Key_Down or event.key() == Qt.Key.Key_S:
            self.scene_pos[1] += 1
        self.update_uv()
        self.signal.key_pressed_signal.emit(self, event)

    def update_uv(self):
        self.real_uv = self.scene_pos
        if self.current_status == "none" or self.current_status == "point":
            pass

        if self.current_status == "line":
            if abs(self.slope) < 0.45:
                u = self.scene_pos[0]
                v = self.slope * u + self.intercept
            else:
                v = self.scene_pos[1]
                u = (v - self.intercept) / (self.slope + 0.000000001)  # 这个斜率截距在display()会自动更新
            self.real_uv = np.array([u, v])
        if self.real_uv is not None:
            # 保留两位小数，使用 f-string 格式化输出
            self.uv_label.setText(f"({self.real_uv[0]:.2f}, {self.real_uv[1]:.2f})")
            self.guide_point_item.set_item_pos(self.real_uv)
            self.guide_point_item.set_item_visibility(True)
        else:
            self.uv_label.setText("(0.00, 0.00)")
            self.guide_point_item.set_item_visibility(False)

    def update_real_uv(self, uv):
        self.real_uv = uv
        self.scene_pos = uv
        self.update_uv()
            
    def activate_self(self):
        self.qt_view.setStyleSheet("border: 3px solid yellow;")

    def deactivate_self(self):
        self.qt_view.setStyleSheet("")

    def draw_pj_line(self, slope, intercept):
        self.slope, self.intercept = slope, intercept
        pts = get_line_in_img(slope, intercept)
        self.pj_line_item.set_item_pos(pts)
        self.pj_line_item.set_item_visibility(True)
        if self.current_status == "none":
            self.current_status = "line"
        elif self.current_status == "point":
            self.current_status = "both"

    def reset_self(self):
        self.current_status = "none"
        self.guide_point_item.set_item_visibility(False)
        self.pj_line_item.set_item_visibility(False)

    def set_real_pin(self, pin_pos, dire_pos):
        self.real_pin_item.set_item_pos(pin_pos)
        self.real_pin_item.set_item_visibility(True)
        self.real_dire_item.set_item_pos(dire_pos)
        self.real_dire_item.set_item_visibility(True)
        self.real_line_item.set_item_pos([pin_pos, dire_pos])
        self.real_line_item.set_item_visibility(True)

    def hide_real_pin(self):
        self.real_pin_item.set_item_visibility(False)
        self.real_dire_item.set_item_visibility(False)
        self.real_line_item.set_item_visibility(False)
