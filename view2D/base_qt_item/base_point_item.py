"""
所有基本的点：
1. 内部一个实心圆
2. 外部一个空心圆
"""
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtWidgets import QGraphicsEllipseItem

from .abstract_base_item import AbstractBaseItem


def draw_outer_circle(diameter, color, thickness) -> QGraphicsEllipseItem:
    # 创建一个 QGraphicsEllipseItem 对象（圆）
    rect = QRectF(0, 0, diameter, diameter)
    circle = QGraphicsEllipseItem(rect)
    # 创建一个 QPen 对象，并设置线宽
    pen = QPen(color, thickness)  # 红色，线宽为 1
    circle.setPen(pen)  # 将 QPen 对象应用到 QGraphicsEllipseItem 上
    circle.setZValue(2)
    circle.setVisible(False)
    return circle


def draw_inner_circle(diameter, color, fill_color, thickness) -> QGraphicsEllipseItem:
    circle = draw_outer_circle(diameter, color, thickness)
    # 设置圆的填充颜色为红色
    circle.setBrush(QBrush(fill_color))
    return circle


class BasePointItem(AbstractBaseItem):
    def __init__(self, qt_scene, item_type: str):
        super().__init__(qt_scene)
        self.item_list: list = self.init_item(item_type)

    def init_item(self, item_type):
        color = None
        fill_color = None
        if item_type == "pin":
            color = QColor(255, 0, 0)
            fill_color = QColor(255, 0, 0)
        elif item_type == "dire":
            color = QColor(0, 0, 255)
            fill_color = QColor(0, 0, 255)
        out_diameter = 6
        in_diameter = 4
        thickness = 0.5
        out_circle = draw_outer_circle(out_diameter, color, thickness)
        in_circle = draw_inner_circle(in_diameter, color, fill_color, thickness)
        self.qt_scene.addItem(out_circle)
        self.qt_scene.addItem(in_circle)
        self.properties = {"pos": None, "out_radius": out_diameter / 2, "in_radius": in_diameter / 2,
                           "color": color, "fill_color": fill_color, "thickness": thickness}
        return [out_circle, in_circle]

    def set_item_visibility(self, visible: bool):
        for circle in self.item_list:
            assert isinstance(circle, QGraphicsEllipseItem)
            circle.setVisible(visible)

    def set_item_pos(self, pos):
        out_circle, in_circle = self.item_list
        assert isinstance(out_circle, QGraphicsEllipseItem)
        assert isinstance(in_circle, QGraphicsEllipseItem)
        out_circle.setPos(pos[0] - self.properties["out_radius"], pos[1] - self.properties["out_radius"])
        in_circle.setPos(pos[0] - self.properties["in_radius"], pos[1] - self.properties["in_radius"])
        self.properties["pos"] = pos

    def set_item_is_highlight(self, highlight: bool):
        color = {True: self.properties["color"], False: QColor(128, 128, 128)}
        fill_color = {True: self.properties["fill_color"], False: QColor(128, 128, 128)}
        assert isinstance(self.item_list[0], QGraphicsEllipseItem)
        assert isinstance(self.item_list[1], QGraphicsEllipseItem)
        self.item_list[0].setPen(QPen(color[highlight], self.properties["thickness"]))
        self.item_list[1].setPen(QPen(color[highlight], self.properties["thickness"]))
        self.item_list[1].setBrush(QBrush(fill_color[highlight]))
