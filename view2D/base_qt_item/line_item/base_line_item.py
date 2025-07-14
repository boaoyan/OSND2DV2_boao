from abc import abstractmethod, ABC
from PyQt5.QtWidgets import QGraphicsLineItem
from ..abstract_base_item import AbstractBaseItem


class BaseLineItem(AbstractBaseItem, ABC):
    def __init__(self, qt_scene):
        super().__init__(qt_scene)
        self.item_list = []

    @abstractmethod
    def init_item(self):
        pass

    def set_item_visibility(self, visible:bool):
        for line in self.item_list:
            assert isinstance(line, QGraphicsLineItem)
            line.setVisible(visible)

    @abstractmethod
    def set_item_pos(self, item_pos:list):
        pass