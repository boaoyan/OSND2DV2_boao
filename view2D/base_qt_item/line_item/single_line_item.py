import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsLineItem

from .base_line_item import BaseLineItem
from .utils import draw_line


class SingleLineItem(BaseLineItem):
    def __init__(self, qt_scene):
        super().__init__(qt_scene)
        self.line_item = self.init_item()

    def init_item(self):
        thickness = 1.5
        style = Qt.PenStyle.SolidLine
        color = QColor(255, 240, 200)
        line = draw_line(color, thickness, style)
        self.qt_scene.addItem(line)
        self.properties = {"start_pt": (0, 0), "end_pt": (1, 1), "color": color,
                           "thickness": thickness, "style": style}
        return line

    def set_item_pos(self, item_pos: list):
        start_pt, end_pt = item_pos

        direction = start_pt - end_pt
        if np.linalg.norm(direction) < 1e-8:
            # 两点重合，无法定义方向
            return start_pt, end_pt
        direction = direction / np.linalg.norm(direction)
        ext1 = start_pt + direction * 10000
        ext2 = end_pt - direction * 10000

        self.line_item.setPen(QPen(self.properties["color"], self.properties["thickness"], self.properties["style"]))
        # self.line_item.setLine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
        self.line_item.setLine(ext1[0], ext1[1], ext2[0], ext2[1])

    def set_item_visibility(self, visible: bool):
        self.line_item.setVisible(visible)
