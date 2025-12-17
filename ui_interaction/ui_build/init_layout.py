from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QSpinBox

from ui.nav_2dv2 import Ui_MainWindow


class InitUILayout(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 设置窗口标题和图标
        self.setWindowTitle("2D穿刺导航界面")
        self.setWindowIcon(QIcon("res/nav_icon.jpg"))





