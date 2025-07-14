"""
规划路径图元：
1. 包含路径线
2. 包含靶点和上点
"""
import numpy as np
from PyQt5.QtWidgets import QGraphicsScene
from .base_point_item import BasePointItem
from .line_item import PathLineItem
from .abstract_base_item import AbstractBaseItem
from typing import Union, List, Tuple

def update_point_item(point_item: BasePointItem, plan_uv_pos: np.ndarray):
    point_item.set_item_visibility(True)
    point_item.set_item_pos(plan_uv_pos)

class NavPathItem(AbstractBaseItem):
    def __init__(self, qt_scene:QGraphicsScene):
        super().__init__(qt_scene)
        # 路径线
        self.path_line_item = PathLineItem(qt_scene)
        # 路径点
        self.pin_item = BasePointItem(qt_scene, "pin")
        self.dire_item = BasePointItem(qt_scene, "dire")

    def init_item(self):
        pass

    def set_item_visibility(self, visible: Union[bool, Tuple[bool, bool, bool], None]):
        if isinstance(visible, bool):
            self.path_line_item.set_item_visibility(visible)
            self.pin_item.set_item_visibility(visible)
            self.dire_item.set_item_visibility(visible)
        elif isinstance(visible, (list, tuple)):
            self.path_line_item.set_item_visibility(visible[0])
            self.pin_item.set_item_visibility(visible[1])
            self.dire_item.set_item_visibility(visible[2])
        else:
            # 智能判断显示，如果靶点和上点的位置很接近，则只显示靶点
            pin_pos = self.pin_item.properties["pos"]
            dire_pos = self.dire_item.properties["pos"]
            if np.linalg.norm(pin_pos - dire_pos) < 1e-3:
                self.set_item_visibility((False, True, False))
            else:
                self.set_item_visibility(True)

    def set_item_pos(self, path_pos:list):
        pin_pos, dire_pos = path_pos[0]
        update_point_item(self.pin_item, pin_pos)
        update_point_item(self.dire_item, dire_pos)
        self.path_line_item.set_item_pos(path_pos)

    def set_item_is_highlight(self, highlight:bool):
        self.path_line_item.set_item_is_highlight(highlight)
        self.pin_item.set_item_is_highlight(highlight)
        self.dire_item.set_item_is_highlight(highlight)
