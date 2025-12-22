import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer

from camera_communication.get_cam_data import UdpReceiverThread
from ui_interaction.ui_build.init_layout import InitUILayout
from ui_interaction.ui_build.voxel_widget_build.voxel_load_widget import VoxelLoadClipWidget
from ui_interaction.ui_response.base_event_type.control_event import ControlEvent
from ui_interaction.ui_response.base_event_type.guide_event import GuideEvent
from view2D.view_manager import ViewerManager
from view2D.view_render import ViewRender
from config import ConfigManager
import cv2 as cv




class InitPara(InitUILayout):
    def __init__(self):
        super().__init__()
        # 导入数据配置文件
        self.data_config = ConfigManager().get_instance().get_all_configs("default_data_path_config")
        # 初始化相机参数
        self.camera_thread = UdpReceiverThread(self.data_config["camera_param"]["ip"],
                                               self.data_config["camera_param"]["port"])
        self.camera_thread.start()
        # 初始化脊柱3D图像
        voxel_path = self.data_config["vox_image"]["vox_img_path"]
        self.voxel_load_clip_ui = VoxelLoadClipWidget(self.voxel_view, voxel_path)

        # 初始化视图显示窗口
        front_img = cv.imread(self.data_config["ct_image"]["front_img_path"], 0)
        self.front_view_render = ViewRender(self.front_view, front_img, self.front_uv_label,
                                            self.data_config["trans_matrix"]["rt_ct2o_sz"])
        side_img = cv.imread(self.data_config["ct_image"]["side_img_path"], 0)
        self.side_view_render = ViewRender(self.side_view, side_img, self.side_uv_label,
                                           self.data_config["trans_matrix"]["rt_ct2o_sc"])
        self.view_manager = ViewerManager()
        self.view_manager.viewers.append(self.front_view_render)
        self.view_manager.viewers.append(self.side_view_render)

        # 添加事件处理
        balls_pts = pd.read_csv(self.data_config["balls_pts"])
        balls_in_ct = balls_pts[["ct_x", "ct_y", "ct_z"]].values
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
        }
        self.control_event = ControlEvent(**control_event_params)
