import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QPushButton

from camera_communication.get_cam_data import UdpReceiverThread
from robot_arm import RobotArmControl
from robot_arm.utils.iterate_move_control import get_distance_to_line
from ui_interaction.ui_response.utils.math_transform import get_pixel_from_ct, get_pj_pt
from ui_interaction.ui_response.utils.visual_funcs import plot_line_and_point, plot_5_points, plot_coordinate_frames
from view2D.view_render import ViewRender


def transform_point(rt_matrix, point):
    """
    将一个三维点与RT矩阵相乘，返回变换后的三维点。

    参数:
        rt_matrix (4x4 numpy array): 齐次变换矩阵（旋转+平移）
        point (3-element list, tuple, or numpy array): 输入的三维点 [x, y, z]

    返回:
        numpy array: 变换后的三维点 [x', y', z']
    """
    # 将三维点转换为齐次坐标 [x, y, z, 1]
    point_homogeneous = np.append(point, 1)

    # 矩阵乘法: RT * point
    transformed_point_homogeneous = rt_matrix @ point_homogeneous

    # 转换回三维坐标（除以w分量，尽管通常为1）
    # 对于仿射变换，w应该接近1，可以直接取前三个分量
    transformed_point = transformed_point_homogeneous[:3]

    return transformed_point


class ControlEvent:
    def __init__(self,init_para, a_arm: str,
                 sz_view_render: ViewRender, sc_view_render: ViewRender,
                 camera_thread: UdpReceiverThread,
                 dire_cam_pos_label: QLabel, pin_cam_pos_label: QLabel,
                 robot_arm_param: dict,
                 connect_device_btn: QPushButton,
                 reset_arm_btn: QPushButton,
                 toggle_pin_pos_order_btn: QPushButton,
                 cali_arm_btn: QPushButton,
                 control_to_aim: QPushButton,
                 fix_error_btn: QPushButton,
                 cali_wait_timer: QTimer,
                 result_visual_timer: QTimer,
                 result_judge_timer:QTimer):
        self.voxel_load_clip_ui = init_para.voxel_load_clip_ui
        self.sz_view_render = sz_view_render
        self.sc_view_render = sc_view_render
        self.camera_thread = camera_thread
        self.camera_thread.data_refreshed.connect(self.update_ui_info)

        # 正侧位视图的投影参数
        self.a_arm = np.load(a_arm)
        self.a_inv = np.linalg.inv(self.a_arm)
        self.L = 800

        self.rt_cam2ct = None

        # ui控件相关
        self.dire_cam_pos_label = dire_cam_pos_label
        self.pin_cam_pos_label = pin_cam_pos_label

        # 机械臂控制相关
        self.arm_control = RobotArmControl(robot_arm_param)
        self.connect_device_btn = connect_device_btn
        self.reset_arm_btn = reset_arm_btn
        self.cali_arm_btn = cali_arm_btn
        self.control_to_aim = control_to_aim

        self.connect_device_btn.clicked.connect(self.connect_serial)
        self.reset_arm_btn.clicked.connect(self.arm_control.reset_arm)

        # 机械臂标定
        self.toggle_pin_pos_order_btn = toggle_pin_pos_order_btn
        self.toggle_pin_pos_order_btn.clicked.connect(self.toggle_pin_pos_order)
        self.cali_arm_btn.clicked.connect(self.cali_arm)
        self.cali_wait_timer = cali_wait_timer
        self.cali_wait_timer.setSingleShot(True)  # 设置为单次触发
        self.cali_wait_timer.timeout.connect(self.cali_arm_again)

        # 机械臂指向目标点
        self.control_to_aim.clicked.connect(self.control_to_aim_func)
        self.aim_in_cam = None
        self.previous_distance = None

        self.result_judge_timer = result_judge_timer
        self.result_judge_timer.setSingleShot(True)
        self.result_judge_timer.timeout.connect(self.result_judge)

        self.result_visual_timer = result_visual_timer
        self.result_visual_timer.setSingleShot(True)  # 设置为单次触发
        self.result_visual_timer.timeout.connect(self.result_visual)
        # 修正误差
        self.fix_error_btn = fix_error_btn
        self.fix_error_btn.clicked.connect(self.fix_error)


    def connect_serial(self):
        self.camera_thread.start_listening()
        self.arm_control.connect_arm_serial()

    def update_real_pin(self):
        if self.rt_cam2ct is None:
            print("rt_cam2ct is None, 无法投影真实针到体素坐标系中")
            return
        if self.camera_thread.pin_balls_in_cam is not None:
            real_dire_in_cam = self.camera_thread.pin_balls_in_cam[1]
            real_pin_in_cam = self.camera_thread.pin_balls_in_cam[0]
            real_dire_in_ct = transform_point(self.rt_cam2ct, real_dire_in_cam)
            real_pin_in_ct = transform_point(self.rt_cam2ct, real_pin_in_cam)

            self.voxel_load_clip_ui.show_pin_in_ct(real_dire_in_ct, real_pin_in_ct)
            # 针尾在两个坐标系下的坐标
            real_dire_uv1, real_dire_uv2 = get_pixel_from_ct(real_dire_in_ct,
                                                             self.sz_view_render.rt_ct2o,
                                                             self.sc_view_render.rt_ct2o,
                                                             self.a_arm)
            # 针尖在两个坐标系下的坐标
            real_pin_uv1, real_pin_uv2 = get_pixel_from_ct(real_pin_in_ct,
                                                           self.sz_view_render.rt_ct2o,
                                                           self.sc_view_render.rt_ct2o,
                                                           self.a_arm)

            self.sz_view_render.set_real_pin(real_pin_uv1, real_dire_uv1)
            self.sc_view_render.set_real_pin(real_pin_uv2, real_dire_uv2)

            self.dire_cam_pos_label.setText(f'({real_dire_in_cam[0]:.2f}, '
                                            f'{real_dire_in_cam[1]:.2f}, '
                                            f'{real_dire_in_cam[2]:.2f})')
            self.pin_cam_pos_label.setText(f'({real_pin_in_cam[0]:.2f}, '
                                           f'{real_pin_in_cam[1]:.2f}, '
                                           f'{real_pin_in_cam[2]:.2f})')

    def update_rt_cam2ct(self, rt_ct2cam):
        self.rt_cam2ct = np.linalg.inv(rt_ct2cam)
        self.update_real_pin()

    def update_ui_info(self):
        if self.camera_thread.pin_balls_in_cam is not None:
            real_dire_in_cam = self.camera_thread.pin_balls_in_cam[0]
            real_pin_in_cam = self.camera_thread.pin_balls_in_cam[1]
            self.dire_cam_pos_label.setText(f'({real_dire_in_cam[0]:.2f}, '
                                            f'{real_dire_in_cam[1]:.2f}, '
                                            f'{real_dire_in_cam[2]:.2f})')
            self.pin_cam_pos_label.setText(f'({real_pin_in_cam[0]:.2f}, '
                                           f'{real_pin_in_cam[1]:.2f}, '
                                           f'{real_pin_in_cam[2]:.2f})')

    def cali_arm(self):
        self.arm_control.current_cali_pos_index = 0
        self.arm_control.balls_in_cam = []
        pos_a = self.arm_control.cali_a_sequence[self.arm_control.current_cali_pos_index]
        pos_b = self.arm_control.cali_b_sequence[self.arm_control.current_cali_pos_index]
        self.arm_control.move(pos_a, pos_b)
        self.cali_wait_timer.start(5000)

    def cali_arm_again(self):
        self.arm_control.current_cali_pos_index += 1
        self.arm_control.balls_in_cam.append(self.camera_thread.pin_balls_in_cam[0])
        self.arm_control.balls_in_cam.append(self.camera_thread.pin_balls_in_cam[1])
        self.arm_control.update_current_pos_in_arm()
        if self.arm_control.current_cali_pos_index >= len(self.arm_control.cali_a_sequence):
            self.arm_control.current_cali_pos_index = 0
            self.arm_control.cali()
            self.cali_wait_timer.stop()
            return
        pos_a = self.arm_control.cali_a_sequence[self.arm_control.current_cali_pos_index]
        pos_b = self.arm_control.cali_b_sequence[self.arm_control.current_cali_pos_index]
        self.arm_control.move(pos_a, pos_b)
        self.cali_wait_timer.start(2000)

    def control_to_aim_func(self):
        if self.aim_in_cam is None:
            print("aim_in_cam is None, 无法控制机械臂指向目标点")
            return
        self.arm_control.control_to_aim(self.aim_in_cam[:3])
        self.result_visual_timer.start(2000)

    def fix_error(self):
        if self.previous_distance < 0.5:
            print("距离小于0.5mm，无需修正")
        else:
            aim_in_cam = self.aim_in_cam[:3]
            P0 = self.camera_thread.pin_balls_in_cam[0]
            P1 = self.camera_thread.pin_balls_in_cam[1]
            pj_pt = get_pj_pt(aim_in_cam, P0, P1)
            aim_in_cam1 = 2*aim_in_cam - pj_pt
            # aim_in_cam2 = aim_in_cam - pj_pt + aim_in_cam
            plot_5_points(P0, P1, aim_in_cam, pj_pt, aim_in_cam1)
            self.arm_control.control_to_aim(aim_in_cam1)

            self.result_judge_timer.start(1000)


    def result_judge(self):
        P0 = self.camera_thread.pin_balls_in_cam[0]
        P1 = self.camera_thread.pin_balls_in_cam[1]
        current_distance = get_distance_to_line(self.aim_in_cam[:3], P0, P1)
        if self.previous_distance is not None:
            difference = current_distance - self.previous_distance
            print('距离差：', difference)

            if difference < 0:
                self.result_visual_timer.start(1000)
            else:
                self.arm_control.move(self.arm_control.true_previous_a,self.arm_control.true_previous_b)
                self.arm_control.previous_optimal_a = self.arm_control.true_previous_a
                self.arm_control.previous_optimal_b = self.arm_control.true_previous_b
                self.result_visual_timer.start(1000)




    def result_visual(self):
        P0 = self.camera_thread.pin_balls_in_cam[0]
        P1 = self.camera_thread.pin_balls_in_cam[1]
        plot_line_and_point(P0, P1, self.aim_in_cam[:3])
        current_distance = get_distance_to_line(self.aim_in_cam[:3], P0, P1)
        print("目标点到投影点的距离：", current_distance)

        self.previous_distance = current_distance
        # print("定位球坐标:", P0, P1)
        # print("目标点坐标:", self.aim_in_cam[:3])

    def toggle_pin_pos_order(self):
        self.camera_thread.toggle_pin_pos_order()


