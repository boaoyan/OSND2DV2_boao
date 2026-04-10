import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from sympy.printing.pytorch import torch

from camera_communication.form_viewer import WaveformWidget
from camera_communication.get_cam_data import UdpReceiverThread
from camera_communication.wave_view.time_series_viewer import DualPointVisualizer
from ui_interaction.ui_build.init_layout import InitUILayout
from ui_interaction.ui_build.voxel_widget_build.voxel_load_widget import VoxelLoadClipWidget
from ui_interaction.ui_response.base_event_type.control_event import ControlEvent
from ui_interaction.ui_response.base_event_type.guide_event import GuideEvent
from ui_interaction.ui_response.utils.reg_rt_transform import reg_rt_update
from view2D.view_manager import ViewerManager
from view2D.view_render import ViewRender
from config import ConfigManager
import cv2 as cv




class InitPara(InitUILayout):
    signal_viz_update = pyqtSignal(object, object, object, object)
    def __init__(self):
        super().__init__()
        # 导入数据配置文件
        self.data_config = ConfigManager().get_instance().get_all_configs("default_data_path_config")
        # 初始化相机参数
        self.camera_thread = UdpReceiverThread(self.data_config["camera_param"]["ip"],
                                               self.data_config["camera_param"]["port"])
        self.camera_thread.start()
        # # 初始化波形显示
        # self.visualizer = None
        # # self.camera_thread.data_ready.connect(self._on_data_ready_main_thread)
        # self.camera_thread.data_ready.connect(self._on_independent_data)
        # 初始化脊柱3D图像
        self.voxel_path = self.data_config["vox_image"]["vox_img_path"]
        self.voxel_load_clip_ui = VoxelLoadClipWidget(self.voxel_view)

        # 初始化视图显示窗口
        voxel_nii_path = self.data_config["vox_image"]["vox_nii_path"]
        # mix_model_path = self.data_config["reg_model"]["mix_model_path"]

        front_model_path = self.data_config["reg_model"]["front_model_path"]
        front_img = cv.imread(self.data_config["ct_image"]["front_img_path"], 0)
        self.front_view_render = ViewRender(self.front_view, front_img, self.front_uv_label,
                                            self.data_config["trans_matrix"]["pose_PA"],
                                            voxel_nii_path, front_model_path)
        # self.front_view_render = ViewRender(self.front_view, front_img, self.front_uv_label,
        #                                     self.data_config["trans_matrix"]["rt_ct2o_sz"])


        side_model_path = self.data_config["reg_model"]["side_model_path"]
        side_img = cv.imread(self.data_config["ct_image"]["side_img_path"], 0)
        self.side_view_render = ViewRender(self.side_view, side_img, self.side_uv_label,
                                           self.data_config["trans_matrix"]["pose_RLAT"],
                                           voxel_nii_path, side_model_path)
        # self.side_view_render = ViewRender(self.side_view, side_img, self.side_uv_label,
        #                                    self.data_config["trans_matrix"]["rt_ct2o_sc"])

        # 体素光源配准矩阵融合
        var_euler_dev = pd.read_csv(self.data_config["reg_variance"]["var_euler_dev"])
        var_euler_dev_tran = pd.read_csv(self.data_config["reg_variance"]["var_euler_dev_tran"])
        print("Available columns:", var_euler_dev.columns.tolist())
        reg_var = var_euler_dev[["a_x", "a_y", "a_z", "t_x", "t_y", "t_z"]].values
        reg_var_tran = var_euler_dev_tran[["a_x", "a_y", "a_z", "t_x", "t_y", "t_z"]].values
        self.reg_var = reg_var
        self.reg_var_tran = reg_var_tran
        self.rt_ct2o_sz, self.rt_ct2o_sc = reg_rt_update(self.front_view_render.rt_ct2o,
                                                         self.side_view_render.rt_ct2o,
                                                         self.reg_var,
                                                         self.reg_var_tran)
        print("加载的转换矩阵rt_ct2o_sz为：", self.rt_ct2o_sz)
        print("加载的转换矩阵rt_ct2o_sc为：", self.rt_ct2o_sc)

        self.view_manager = ViewerManager()
        self.view_manager.viewers.append(self.front_view_render)
        self.view_manager.viewers.append(self.side_view_render)

        self.import_img_btn.clicked.connect(self.import_img)
        self.import_voxel_model_btn.clicked.connect(self.update_voxel_model)

        # 添加事件处理
        balls_pts = pd.read_csv(self.data_config["balls_pts"])
        balls_in_ct = balls_pts[["ct_x", "ct_y", "ct_z"]].values
        balls_in_ct = balls_in_ct + np.array([90, 90, -53])
        vox_space = self.data_config["vox_space"]
        guide_event_params = {
            "init_para": self,
            "view_manager": self.view_manager,
            "sz_view_render": self.front_view_render,
            "sc_view_render": self.side_view_render,
            "start_gui_btn": self.start_gui_btn,
            "finish_gui_btn": self.finish_gui_btn,
            "cancel_gui_btn": self.cancel_gui_btn,
            "a_arm": self.data_config["trans_matrix"]["a_arm"],
            "ct_pos_label": self.ct_pos_label,
            "world_aim_pos_label": self.world_aim_pos_label,
            "balls_in_ct": balls_in_ct,
            "vox_space": vox_space,
        }
        self.guide_event = GuideEvent(**guide_event_params)
        # 控制机械臂相关类

        control_event_params = {
            "init_para": self,
            "a_arm": self.data_config["trans_matrix"]["a_arm"],
            "sz_view_render": self.front_view_render,
            "sc_view_render": self.side_view_render,
            "camera_thread": self.camera_thread,
            "dire_cam_pos_label": self.dire_cam_pos_label,
            "pin_cam_pos_label": self.pin_cam_pos_label,
            "robot_arm_param": self.data_config["robot_arm_param"],
            "connect_device_btn": self.connect_device_btn,
            "reset_arm_btn": self.reset_arm_btn,
            "toggle_pin_pos_order_btn": self.toggle_pin_pos_order_btn,
            "cali_arm_btn": self.cali_arm_btn,
            "control_to_aim": self.control_to_aim,
            "fix_error_btn": self.fix_error_btn,
            "cali_wait_timer": QTimer(self),
            "result_visual_timer": QTimer(self),
            "result_judge_timer": QTimer(self),
            "reg_model_btn": self.reg_model_btn,
        }
        self.control_event = ControlEvent(**control_event_params)

    def import_img(self):
        self.front_view_render.show_img()
        self.side_view_render.show_img()
        # self.voxel_load_clip_ui.show_spine(self.voxel_path)


    def update_voxel_model(self):
        try:
            threshold = float(self.spine_threshold.text())  # ← 转为 float
        except ValueError:
            QMessageBox.warning(self, "输入错误", "阈值必须是有效数字！")
            return
        self.voxel_load_clip_ui.show_spine(self.voxel_path, threshold=threshold)



    def update_waveform(self):
        if not self.camera_thread.raw_history:
            return
        print("First raw entry:", self.camera_thread.raw_history[0])
        print("Shape:", np.array(self.camera_thread.raw_history).shape)
        self.waveform_widget.update_plot(
            self.camera_thread.raw_history,
            self.camera_thread.filtered_history,
            pin_order_flipped=getattr(self.camera_thread, 'pin_order_flipped', False)
        )

    def _on_data_ready_main_thread(self, measurement, filtered_state):
        """
        在主线程中接收子线程数据（线程安全）
        measurement, filtered_state: np.ndarray, shape (2, 3)
        """
        # 调试输出
        # print(f"📥 主线程收到数据 | shape: {measurement.shape}")

        # ✅ 通过主线程信号转发到可视化器
        self.signal_viz_update.emit(measurement, filtered_state)

    def set_visualizer(self, viz_window):
        """
        设置独立可视化窗口引用
        由 main.py 调用，建立信号连接
        """
        self.viz_window = viz_window
        # ✅ 连接主线程信号到可视化器
        self.signal_viz_update.connect(self.viz_window.update_data)
        print("✅ 可视化窗口已连接")



    def _on_independent_data(self, measurement, filtered_state):
        """
        接收独立滤波数据 → 运行滤波 → 直接发送到可视化器（只更新模块 1）
        """
        if self.viz_window:
            self.viz_window.update_data(
                measurement_m1=measurement,
                filtered_m1=filtered_state,
                measurement_m2=None,  # 不更新模块 2
                filtered_m2=None
            )

    def _on_correlated_data(self, qua_ct2cam, qua_ct2cam_filter):
        """
        接收模版滤波数据 → 直接发送到可视化器（只更新模块 2）
        """

        # ✅ 直接发送到可视化器（只更新模块 2，模块 1 传 None）
        if self.viz_window:
            self.viz_window.update_data(
                measurement_m1=None,  # 不更新模块 1
                filtered_m1=None,
                measurement_m2=qua_ct2cam,
                filtered_m2=qua_ct2cam_filter
            )
