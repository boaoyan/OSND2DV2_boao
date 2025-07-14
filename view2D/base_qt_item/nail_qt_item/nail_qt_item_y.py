"""
沿x轴投影，在钉子视图应该显示为两个矩形拼接
"""
from PyQt5.QtWidgets import QGraphicsScene

from .abstract_nail_qt_item import AbstractNailQtItem
from .common_nail_paint_funcs import draw_nail_circle


class NailQtItemY(AbstractNailQtItem):
    def __init__(self, qt_scene: QGraphicsScene, **kwargs):
        super().__init__(qt_scene, **kwargs)
        self.init_item()

    def init_item(self):
        self.hat, self.pin = draw_nail_circle(self.outer_diameter,
                                              self.inner_diameter,
                                              [0, 0],
                                              self.hat_style,
                                              self.pin_style)
        super().init_item()

    def set_item_pos(self, pos):
        radius = self.outer_diameter / 2
        self.hat.setPos(0, 0)
        self.hat.setPos(pos[0] - radius, pos[1] - radius)
        radius = self.inner_diameter / 2
        self.pin.setPos(pos[0] - radius, pos[1] - radius)