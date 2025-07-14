import numpy as np
from PyQt5.QtWidgets import QFileDialog

from ui_interaction.ui_build.init_para import InitPara


class EventHandler(InitPara):
    def __init__(self):
        super().__init__()
        self.front_view_render.signal.widget_clicked.connect(self.update_activated_view)
        self.side_view_render.signal.widget_clicked.connect(self.update_activated_view)
        # 键盘事件
        self.front_view_render.signal.key_pressed_signal.connect(self.key_pressed_event)
        self.side_view_render.signal.key_pressed_signal.connect(self.key_pressed_event)
        # 保存和导入规划
        self.save_nav_pos_action.triggered.connect(self.save_nav_pos_triggered)
        self.load_nav_pos_action.triggered.connect(self.load_nav_pos_triggered)

    def showEvent(self, a0):
        self.view_manager.resize_event()

    def resizeEvent(self, a0):
        self.view_manager.resize_event()

    def update_activated_view(self, activated_view):
        self.guide_event.update_activated_view(activated_view)

    def key_pressed_event(self, view, key_event):
        self.guide_event.update_activated_view(view)

    def save_nav_pos_triggered(self):
        if self.guide_event.save_planning():
            return
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "选择一个.npy文件", "output", "NumPy Files (*.npy)",
                                                   options=options)

        if file_name:
            np.save(file_name, self.guide_event.saved_ct_coords)

    def load_nav_pos_triggered(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "选择一个.npy文件", "output", "NumPy Files (*.npy)",
                                                   options=options)

        if file_name:
            self.guide_event.load_planning(file_name)