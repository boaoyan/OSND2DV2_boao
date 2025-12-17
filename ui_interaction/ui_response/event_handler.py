import numpy as np
from PyQt5.QtWidgets import QFileDialog

from ui_interaction.ui_build.init_para import InitPara


class EventHandler(InitPara):
    def __init__(self):
        super().__init__()

        # 更新体素到相机的矩阵
        self.cali_ct2cam_btn.clicked.connect(self.update_rt_ct2cam)
        self.front_view_render.signal.widget_clicked.connect(self.update_activated_view)
        self.side_view_render.signal.widget_clicked.connect(self.update_activated_view)
        # 键盘事件
        self.front_view_render.signal.key_pressed_signal.connect(self.key_pressed_event)
        self.side_view_render.signal.key_pressed_signal.connect(self.key_pressed_event)
        # 保存和导入规划
        self.save_nav_pos_action.triggered.connect(self.save_nav_pos_triggered)
        self.load_nav_pos_action.triggered.connect(self.load_nav_pos_triggered)

        # FIXME: 测试代码
        self.camera_thread.data_refreshed.connect(self.update_rt_ct2cam)

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



    def update_rt_ct2cam(self):
        if self.camera_thread.ct_balls_in_cam is None:
            # print("体素定位球在相机坐标点的位置不存在")
            return
        self.guide_event.update_rt_ct2cam(self.camera_thread.ct_balls_in_cam)
        self.control_event.update_rt_cam2ct(self.guide_event.rt_ct2cam)
        self.control_event.aim_in_cam = self.guide_event.res_w