import numpy as np
from PyQt5.QtWidgets import QPushButton, QLabel

from ui_interaction.ui_build.CompositeControl.message_box import messageBox
from ui_interaction.ui_response.utils.math_transform import get_line, get_coord_in_ct, get_pixel_from_ct, \
    get_point_in_ct
from ui_interaction.ui_response.utils.registration_algorithm import kabsch_numpy
from view2D.view_manager import ViewerManager
from view2D.view_render import ViewRender


class GuideEvent:
    def __init__(self, init_para, view_manager: ViewerManager,
                 start_gui_btn: QPushButton, finish_gui_btn: QPushButton, cancel_gui_btn: QPushButton,
                 a_arm: str, sz_view_render: ViewRender, sc_view_render: ViewRender,
                 ct_pos_label: QLabel, world_aim_pos_label: QLabel, balls_in_ct, vox_space):
        self.voxel_load_clip_ui = init_para.voxel_load_clip_ui
        self.view_manager = view_manager
        self.sz_view_render = sz_view_render
        self.sc_view_render = sc_view_render

        self.start_gui_btn = start_gui_btn
        self.finish_gui_btn = finish_gui_btn
        self.cancel_gui_btn = cancel_gui_btn

        # 状态标志：是否处于“规划中”状态
        self._is_planning = False

        # 绑定按钮点击事件
        self.start_gui_btn.clicked.connect(self.start_plan)
        self.start_gui_btn.setEnabled(True)
        self.finish_gui_btn.clicked.connect(self.finish_plan)
        self.finish_gui_btn.setEnabled(False)
        self.cancel_gui_btn.clicked.connect(self.cancel_plan)
        self.cancel_gui_btn.setEnabled(False)

        # 正侧位视图的投影参数
        self.a_arm = np.load(a_arm)
        self.a_inv = np.linalg.inv(self.a_arm)
        self.L = 800

        self._saved_ct_coords = None
        self.current_ct_coords = None
        self.ct_pos_label = ct_pos_label

        self.balls_in_ct = balls_in_ct
        self.vox_space = np.array(vox_space)
        self.rt_ct2cam = None
        self.res_w = None
        self.world_aim_pos_label = world_aim_pos_label

    @property
    def is_planning(self):
        return self._is_planning

    @is_planning.setter
    def is_planning(self, value: bool):
        if self._is_planning == value:
            return  # 值未变化，无需处理

        self._is_planning = value

        # 根据状态更新 UI 控件
        self.start_gui_btn.setEnabled(not self._is_planning)
        self.finish_gui_btn.setEnabled(self._is_planning)
        self.cancel_gui_btn.setEnabled(self._is_planning)

    @property
    def saved_ct_coords(self):
        return self._saved_ct_coords

    @saved_ct_coords.setter
    def saved_ct_coords(self, value):
        self._saved_ct_coords = value
        self.update_guide_pos_in_cam()

    def update_guide_pos_in_cam(self):
        if self._saved_ct_coords is not None:
            res_w = self.rt_ct2cam @ np.append(self._saved_ct_coords, 1).T
            # print("world aim pos: ", res_w)
            self.world_aim_pos_label.setText(f'({res_w[0]:.2f}, '
                                             f'{res_w[1]:.2f}, '
                                             f'{res_w[2]:.2f})')
            self.res_w = res_w

    def save_planning(self):
        if self.saved_ct_coords is None:
            messageBox("当前没有保存的坐标，无法保存")
            return True

    def load_planning(self, file_path: str):
        if self.is_planning:
            if not messageBox("当前正在规划中，是否丢弃当前规划并加载新的坐标"):
                return True
        if self.saved_ct_coords is not None:
            if not messageBox("加载会覆盖已有规划，是否确认"):
                return True
        self.saved_ct_coords = np.load(file_path)
        self.cancel_plan()

    def start_plan(self):
        """激活“规划中”状态"""
        self.voxel_load_clip_ui.clear_all_guide_lines()
        self.is_planning = True
        self.cancel_gui_btn.setEnabled(True)
        self.reset_plan_views()

    def reset_plan_views(self):
        self.sz_view_render.reset_self()
        self.sc_view_render.reset_self()

    def cancel_plan(self):
        """取消“规划中”状态"""
        if self.is_planning:
            if not messageBox("确定退出，当前的规划将丢失"):
                return
        self.update_plan_view_real_uv()

    def update_plan_view_real_uv(self):
        self.is_planning = False
        self.view_manager.deactivate_all()
        if self.saved_ct_coords is not None:
            self.ct_pos_label.setText(f'({self.saved_ct_coords[0]:.2f}, '
                                      f'{self.saved_ct_coords[1]:.2f}, '
                                      f'{self.saved_ct_coords[2]:.2f})')
            real_uv1, real_uv2 = get_pixel_from_ct(self.saved_ct_coords,
                                                   self.sz_view_render.rt_ct2o,
                                                   self.sc_view_render.rt_ct2o,
                                                   self.a_arm)
            self.reset_plan_views()
            self.sz_view_render.update_real_uv(real_uv1)
            self.sc_view_render.update_real_uv(real_uv2)

    def finish_plan(self):
        if self.sz_view_render.real_uv is not None and self.sc_view_render.real_uv is not None:
            if self.saved_ct_coords is None:
                self.saved_ct_coords = self.current_ct_coords
                self.update_plan_view_real_uv()
            elif messageBox("规划已完成，是否保存并覆盖之前的规划"):
                self.saved_ct_coords = self.current_ct_coords
                self.update_plan_view_real_uv()

    def update_activated_view(self, activated_view):
        if not self.is_planning:
            return

        self.view_manager.update_activated_view(activated_view)

        # 判断激活的是哪个视图
        if activated_view is self.sz_view_render:
            src_view = self.sz_view_render
            dst_view = self.sc_view_render
            view_type = 'sz'
            line_color = 'red'
        elif activated_view is self.sc_view_render:
            src_view = self.sc_view_render
            dst_view = self.sz_view_render
            view_type = 'sc'
            line_color = 'blue'
        else:
            return  # 激活的不是已知视图，不处理

        uv = src_view.real_uv
        if uv is None:
            return  # 没有有效坐标，不处理

        u, v = uv
        oct_source, pot = get_point_in_ct(u, v, src_view.rt_ct2o, self.a_inv, self.L)
        self.voxel_load_clip_ui.show_line_in_ct(oct_source, pot, view_type, line_color)
        slope, intercept = get_line(u, v,
                                    src_view.rt_ct2o, dst_view.rt_ct2o,
                                    self.a_arm, self.a_inv, self.L)
        dst_view.draw_pj_line(slope, intercept)
        # 如果两个点都确定了，则可以返回实际CT体素坐标
        if src_view.real_uv is not None and dst_view.real_uv is not None:
            self.current_ct_coords = get_coord_in_ct(self.sz_view_render.real_uv,
                                                     self.sc_view_render.real_uv,
                                                     self.sz_view_render.rt_ct2o,
                                                     self.sc_view_render.rt_ct2o,
                                                     self.a_inv,
                                                     self.L)
            self.voxel_load_clip_ui.show_selected_point(self.current_ct_coords)
            if self.current_ct_coords is not None:
                self.ct_pos_label.setText(f'({self.current_ct_coords[0]:.2f}, '
                                          f'{self.current_ct_coords[1]:.2f}, '
                                          f'{self.current_ct_coords[2]:.2f})')

    def update_rt_ct2cam(self, balls_in_cam):
        self.rt_ct2cam = kabsch_numpy(self.balls_in_ct, balls_in_cam)
        self.update_guide_pos_in_cam()
