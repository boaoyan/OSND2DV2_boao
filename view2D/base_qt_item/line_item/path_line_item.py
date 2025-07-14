"""
绘制路径线，和投影线不同，
路径线有三段：
1. 靶点和上点之间的连线
2. 靶点的反向延长线
3. 上点的反向延长线
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen

from .base_line_item import BaseLineItem
from .utils import draw_line

class PathLineItem(BaseLineItem):
    def __init__(self, qt_scene):
        super().__init__(qt_scene)
        self.item_list: list = self.init_item()

    def init_item(self):
        line_1 = self.draw_part_of_line("line_1", QColor(255, 192, 203), Qt.PenStyle.SolidLine)
        line_2 = self.draw_part_of_line("line_2", QColor(255, 0, 0), Qt.PenStyle.DashLine)
        line_3 = self.draw_part_of_line("line_3", QColor(0, 0, 255), Qt.PenStyle.DashLine)
        return [line_1, line_2, line_3]

    def draw_part_of_line(self, name:str, color, style):
        thickness = 1
        line = draw_line(color, thickness, style)
        self.qt_scene.addItem(line)
        line_properties = {"start_pt": (0, 0), "end_pt": (1, 1), "color": color,
                           "thickness": thickness, "style": style}
        self.properties = {name: line_properties}
        return line

    def set_item_pos(self, item_pos: list):
        line_names = ["line_1", "line_2", "line_3"]
        for i, line_name in enumerate(line_names):
            start_pt, end_pt = item_pos[i]
            line = self.item_list[i]
            line.setLine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
            self._properties[line_name]["start_pt"] = start_pt
            self._properties[line_name]["end_pt"] = end_pt

    def set_item_is_highlight(self, highlight: bool):
        """
        该函数用于选择是否高亮显示路径线，不会改变其本身内存的properties
        :param highlight: 高亮就是原本的颜色，不高亮就是灰色
        :return:
        """
        line_names = ["line_1", "line_2", "line_3"]
        for line_name, line_item in zip(line_names, self.item_list):
            # 根据 highlight 状态设置颜色
            color_map = {True: self._properties[line_name]["color"], False: QColor(128, 128, 128)}
            line_props = self._properties[line_name]
            color = color_map[highlight]
            line_item.setPen(QPen(color, line_props["thickness"], line_props["style"]))


