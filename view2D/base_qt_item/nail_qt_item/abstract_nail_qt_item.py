"""
定义在2D钉子视图中绘制钉子图元的抽象基类
"""
from abc import ABC

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsScene

from ..abstract_base_item import AbstractBaseItem


class AbstractNailQtItem(AbstractBaseItem, ABC):
    def __init__(self, qt_scene:QGraphicsScene, **kwargs):
        super().__init__(qt_scene)
        # 设置钉子的尺寸
        self.outer_diameter = kwargs.get("outer_diameter", 20)
        self.inner_diameter = kwargs.get("inner_diameter", 10)
        self.pin_length = kwargs.get("pin_length", 120)
        self.hat_length = kwargs.get("hat_length", 20)
        self.pin_style = {"color": Qt.GlobalColor.green, "line_style": Qt.PenStyle.DashLine, "thickness": 1}
        self.hat_style = {"color": Qt.GlobalColor.blue, "line_style": "实线", "thickness": 2}
        self.properties = {"pin_style": self.pin_style, "hat_style": self.hat_style}
        self.pin = None
        self.hat = None

    def init_item(self):
        self.hat.setZValue(1)
        self.pin.setZValue(1)
        self.qt_scene.addItem(self.hat)
        self.qt_scene.addItem(self.pin)

    def set_item_visibility(self, visible:bool):
        self.pin.setVisible(visible)
        self.hat.setVisible(visible)
