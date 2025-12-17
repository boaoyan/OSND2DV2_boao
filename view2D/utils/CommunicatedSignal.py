# 定义一个带有自定义信号的类
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QKeyEvent


class Communicate(QObject):
    widget_clicked = pyqtSignal(object)  # 当点击某个视图时，发出该信号，将被点击的视图实例传递
    widget_double_clicked = pyqtSignal(object)  # 当双击某个视图时，发出该信号，将被双击的视图实例传递
    wheel_scrolled_signal = pyqtSignal(object, int)  # 当滚动鼠标滚轮时，发出该信号，将滚动的距离传递
    mouse_enter_signal = pyqtSignal(object)  # 当鼠标进入某个视图时，发出该信号，将进入的视图实例传递
    mouse_leave_signal = pyqtSignal(object)  # 当鼠标离开某个视图时，发出该信号，将离开的视图实例传递
    key_pressed_signal = pyqtSignal(object, QKeyEvent)  # 当按下某个键盘按键时，发出该信号，将该视图实例和按下的键盘按键传递
    str_update_signal = pyqtSignal(str)