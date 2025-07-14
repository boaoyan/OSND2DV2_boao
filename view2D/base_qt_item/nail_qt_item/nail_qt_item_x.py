"""
沿x轴投影，在钉子视图应该显示为两个矩形拼接
"""
import numpy as np
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QGraphicsScene
from .abstract_nail_qt_item import AbstractNailQtItem
from .common_nail_paint_funcs import draw_nail_rect

class NailQtItemX(AbstractNailQtItem):
    def __init__(self, qt_scene:QGraphicsScene, **kwargs):
        super().__init__(qt_scene, **kwargs)
        self.init_item()

    def init_item(self):
        hat_pos = [self.pin_length, 0]
        hat_width = self.hat_length
        hat_height = self.outer_diameter
        pin_pos = [0, (self.outer_diameter - self.inner_diameter) / 2]
        pin_width = self.pin_length
        pin_height = self.inner_diameter
        hat_size = np.array([hat_width, hat_height])
        pin_size = np.array([pin_width, pin_height])
        self.hat, self.pin = draw_nail_rect(hat_pos, hat_size, self.hat_style,
                                            pin_pos, pin_size, self.pin_style)
        super().init_item()

    def set_item_pos(self, pos):
        hat_pos = QPointF(pos[0], pos[1] - self.outer_diameter / 2)
        pin_pos = QPointF(pos[0], pos[1] - self.outer_diameter / 2)
        self.hat.setPos(hat_pos)
        self.pin.setPos(pin_pos)