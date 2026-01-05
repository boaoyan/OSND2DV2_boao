import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtSerialPort import QSerialPort

from robot_arm.utils.cali_utils import display_cali_result, transform_points
from robot_arm.utils.iterate_move_control import IterateMoveControl

from robot_arm.utils.serial_utils import print_available_ports
from robot_arm.utils.tool_to_arm import get_tool2arm_rt
from ui_interaction.ui_response.utils.registration_algorithm import kabsch_numpy


class RobotArmControl(QThread):
    # 可选：添加信号用于通知连接状态或接收数据
    connected = pyqtSignal(bool)
    data_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.serial = QSerialPort(self)
        self.serial.setPortName(config['port_name'])
        self.serial.setBaudRate(config['baudrate'])

        # 确保使用正确的枚举类型
        self.serial.setDataBits(QSerialPort.Data8)  # 或根据需要调整
        self.serial.setParity(QSerialPort.NoParity)  # 或根据需要调整
        self.serial.setStopBits(QSerialPort.OneStop)  # 或根据需要调整

        # 机械臂初始位置
        self.init_a = config['init_a']
        self.init_b = config['init_b']
        self.reset_a = config['reset_a']
        self.reset_b = config['reset_b']
        # 两个机械臂的限位
        self.a_threshold = config['a_threshold']
        self.b_threshold = config['b_threshold']
        # 机械臂参数
        self.param_toolP = np.array(config['param_toolP'])
        self.param_cz = np.array(config['param_cz'])
        # 机械臂标定
        # cali_a = config['cali_a']
        # cali_b = config['cali_b']
        # cali_num = config['cali_num']
        # self.cali_a_sequence = np.linspace(cali_a[0], cali_a[1], cali_num)
        # self.cali_b_sequence = np.linspace(cali_b[0], cali_b[1], cali_num)
        cali_a_new = config['cali_a_new']
        cali_b_new = config['cali_b_new']
        self.cali_a_sequence = np.array(cali_a_new)
        self.cali_b_sequence = np.array(cali_b_new)
        self.current_cali_pos_index = 0
        self.rt_arm2cam = None
        self.previous_optimal_a = None
        self.previous_optimal_b = None
        self.true_previous_a = None
        self.true_previous_b = None
        self.balls_in_arm = []
        self.balls_in_cam = []
        # 测试运动控制
        self.param_R2C = np.array(config['param_R2C'])
        self.iterate_move_control = IterateMoveControl(config)

    def reset_arm(self):
        self.move(self.reset_a, self.reset_b)

    def connect_arm_serial(self):
        """
        打开并连接串口
        :return: bool, 是否成功
        """
        if self.serial.isOpen():
            print("串口已打开")
            self.connected.emit(True)
            return True

        if self.serial.open(QSerialPort.ReadWrite):
            print(f"✅ 成功连接到串口 {self.serial.portName()}")
            # 连接成功移动到中位
            self.reset_arm()
            self.connected.emit(True)
            return True
        else:
            error_msg = self.serial.errorString()
            port_name = self.serial.portName()
            print(f"❌ 无法打开串口 {port_name}: {error_msg}")

            # 打印当前可用端口信息
            print_available_ports()

            self.error_occurred.emit(f"串口错误: {error_msg}")
            self.connected.emit(False)
            return False

    def disconnect_serial(self):
        """
        关闭串口
        """
        if self.serial.isOpen():
            self.serial.close()
            print("串口已关闭")
        else:
            print("串口未打开")

    def move(self, a, b):
        a = max(self.a_threshold[0], min(a, self.a_threshold[1]))
        b = max(self.b_threshold[0], min(b, self.b_threshold[1]))
        command = f"MOVE {a} {b} 300 300\n"
        print("机械臂运动位置:", command)
        self.serial.write(command.encode())

    def read_serial_data(self):
        # 等待串口数据一段时间
        if self.serial.waitForReadyRead(2000):  # 等待最多 2 秒
            data = self.serial.readAll().data().decode()
        else:
            data = "Error: Timeout waiting for data"
        return data

    def cali(self):
        balls_in_arm = np.array(self.balls_in_arm)
        balls_in_cam = np.array(self.balls_in_cam)
        self.rt_arm2cam = kabsch_numpy(balls_in_arm, balls_in_cam)
        print("成功标定相机位置")
        print(self.rt_arm2cam)
        self.iterate_move_control.rt_arm2cam = self.rt_arm2cam
        display_cali_result(balls_in_arm, balls_in_cam, self.rt_arm2cam)
        self.reset_arm()
        self.balls_in_arm = []
        self.balls_in_cam = []

    def update_current_pos_in_arm(self):
        """
        更新当前机械臂在当前ab对应定位球在底座坐标系的位置
        :return:
        """
        command = "GET\n"
        self.serial.write(command.encode())
        data = self.read_serial_data()
        if data:
            lines = data.splitlines()
            if len(lines) >= 2:
                a_pos = lines[0].strip()
                b_pos = lines[1].strip()
                print("机械臂实际位置：", a_pos, b_pos)
                a_pos = ((float(a_pos) - self.init_a) / 2048) * np.pi
                b_pos = ((float(b_pos) - self.init_b) / 2048) * np.pi
                tool2arm_r, tool2arm_t = get_tool2arm_rt(a_pos, b_pos, self.param_cz)
                param_toolP = self.param_toolP.T
                arm_pos = tool2arm_r @ param_toolP + tool2arm_t
                arm_pos = arm_pos.T
                self.balls_in_arm.append(arm_pos[0])
                self.balls_in_arm.append(arm_pos[1])
            else:
                print("电机位置：解析失败")
        else:
            print("电机位置：超时或无响应")

    def control_to_aim(self, aim_in_cam):
        self.true_previous_a = self.previous_optimal_a
        self.true_previous_b = self.previous_optimal_b
        print("目标点", aim_in_cam)
        # a_sol, b_sol = goto(self.param_toolP, self.param_cz,
        #                     np.linalg.inv(self.rt_arm2cam), aim_in_cam[:3])
        # a_sol, b_sol = goto_test(self.param_toolP, self.param_cz,
        #                          self.param_R2C, aim_in_cam[:3])
        # a = round(((a_sol % (2 * np.pi)) / np.pi) * 2048)
        # b = round(((b_sol % (2 * np.pi)) / np.pi) * 2048)
        # a += self.init_a
        # b += self.init_b
        x0 = [self.init_a, self.init_b]
        optimal_a, optimal_b = self.iterate_move_control.control_to_aim(aim_in_cam, x0)
        print("最优解", optimal_a, optimal_b)
        self.move(optimal_a, optimal_b)
        self.previous_optimal_a = optimal_a
        self.previous_optimal_b = optimal_b


