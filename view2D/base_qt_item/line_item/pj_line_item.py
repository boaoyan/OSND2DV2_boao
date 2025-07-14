from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsLineItem

from .base_line_item import BaseLineItem
from .utils import draw_line


class PJLineItem(BaseLineItem):
    def __init__(self, qt_scene):
        super().__init__(qt_scene)
        self.item_list: list = self.init_item()

    def init_item(self):
        thickness = 0.5
        style = Qt.PenStyle.DashLine
        color = QColor(255, 240, 200)
        line = draw_line(color, thickness, style)
        self.qt_scene.addItem(line)
        self.properties = {"start_pt": (0, 0), "end_pt": (1, 1), "color": color,
                           "thickness": thickness, "style": style}
        return [line]

    def set_item_pos(self, item_pos: list):
        start_pt, end_pt = item_pos
        line = self.item_list[0]
        assert isinstance(line, QGraphicsLineItem)
        line.setPen(QPen(self.properties["color"], self.properties["thickness"], self.properties["style"]))
        line.setLine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
