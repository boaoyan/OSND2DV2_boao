"""
qt_render中绘制的所有图形的基类
1、 需要告知该图元和哪个qt_render绑定
2、可以设置每个图元的属性
"""

from abc import ABC, abstractmethod
from PyQt5.QtWidgets import QGraphicsScene


class AbstractBaseItem(ABC):
    def __init__(self, qt_scene:QGraphicsScene):
        self.qt_scene = qt_scene
        self._properties = {}

    @property
    def properties(self):
        return self._properties

    @properties.setter
    def properties(self, _properties:dict):
        for key, value in _properties.items():
            self._properties[key] = value

    @abstractmethod
    def init_item(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_item_visibility(self, visible: bool):
        pass

    @abstractmethod
    def set_item_pos(self, pos):
        pass