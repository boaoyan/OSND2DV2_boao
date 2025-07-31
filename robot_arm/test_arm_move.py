import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QSlider, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtSerialPort import QSerialPort
from camera_communication.get_cam_data import UdpReceiverThread

# 参数（未使用，保留）
param_toolP = [[37.94320527,  5.06780067, 44.40407641],
               [-36.66520007,  -6.29309517, -42.13093802]]
param_R2C = [4.6341918306127985, 66.95422191178551, -518.1065302867709,    0.5803091298116838, 0.8179104349174249, -0.2676898490602819]
param_cz = [0.01432229, 1.23184338]

def arm_pos_to_world(a, b):
    c, z, = param_cz
    Tx, Ty, Tz, A, B, C = param_R2C

    sc = np.sin(c)
    cc = np.cos(c)
    sA = np.sin(A)
    cA = np.cos(A)
    sB = np.sin(B)
    cB = np.cos(B)
    sC = np.sin(C)
    cC = np.cos(C)

    sa = np.sin(a)
    ca = np.cos(a)
    sb = np.sin(b)
    cb = np.cos(b)
    # print("1:", Tx, Ty, Tz, X1, Y1, Z1, X2, Y2, Z2, z, c)
    X0, Y0, Z0 = param_toolP[0]
    X1, Y1, Z1 = param_toolP[1]
    arm_p0_in_cam = arm_to_world(Tx, Ty, Tz, X0, Y0, Z0, cA, cB, cC, ca, cb, cc, sA, sB, sC, sa, sb, sc, z)
    arm_p1_in_cam = arm_to_world(Tx, Ty, Tz, X1, Y1, Z1, cA, cB, cC, ca, cb, cc, sA, sB, sC, sa, sb, sc, z)
    # print(x[0],y[0],Z)
    return arm_p0_in_cam, arm_p1_in_cam


def arm_to_world(Tx, Ty, Tz, X1, Y1, Z1, cA, cB, cC, ca, cb, cc, sA, sB, sC, sa, sb, sc, z):
    x = X1 * (cb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) + sb * (sB * ca - cB * sC * sa)) - Tx - Z1 * (
            sb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) - cb * (sB * ca - cB * sC * sa)) - Y1 * (
                cc * (sB * sa + cB * sC * ca) - cB * cC * sc) - z * (sB * ca - cB * sC * sa),
    y = Y1 * (sc * (cA * sC + cC * sA * sB) + cc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) - z * (
            sa * (cA * cC - sA * sB * sC) - cB * sA * ca) - Ty + X1 * (cb * (
            cc * (cA * sC + cC * sA * sB) - sc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) + sb * (
                                                                               sa * (
                                                                               cA * cC - sA * sB * sC) - cB * sA * ca)) - Z1 * (
                sb * (cc * (cA * sC + cC * sA * sB) - sc * (
                ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) - cb * (
                        sa * (cA * cC - sA * sB * sC) - cB * sA * ca)),
    Z = Y1 * (cc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa) + sc * (sA * sC - cA * cC * sB)) - z * (
            sa * (cC * sA + cA * sB * sC) + cA * cB * ca) - Tz + X1 * (
                sb * (sa * (cC * sA + cA * sB * sC) + cA * cB * ca) + cb * (
                cc * (sA * sC - cA * cC * sB) - sc * (
                ca * (cC * sA + cA * sB * sC) - cA * cB * sa))) - Z1 * (sb * (
            cc * (sA * sC - cA * cC * sB) - sc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa)) - cb * (
                                                                                sa * (
                                                                                cC * sA + cA * sB * sC) + cA * cB * ca))
    return x[0], y[0], Z

def compute_Tend(alpha, beta):
    Tx, Ty, Tz, A, B, C = param_R2C
    c, z = param_cz
    # print(Tx,Ty,Tz,A,B,C,c,z)
    # c,z = 0.1,0.5
    sc = np.sin(c)
    cc = np.cos(c)
    sA = np.sin(A)
    cA = np.cos(A)
    sB = np.sin(B)
    cB = np.cos(B)
    sC = np.sin(C)
    cC = np.cos(C)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    sb = np.sin(beta)
    cb = np.cos(beta)
    # print(sA,cA,sB,cB,sC,cC)
    return np.array([
        [
            cb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc) + sb * (sB * ca - cB * sC * sa),
            cB * cC * sc - cc * (sB * sa + cB * sC * ca),
            cb * (sB * ca - cB * sC * sa) - sb * (sc * (sB * sa + cB * sC * ca) + cB * cC * cc)
        ],
        [
            cb * (cc * (cA * sC + cC * sA * sB) - sc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)) + sb * (
                        sa * (cA * cC - sA * sB * sC) - cB * sA * ca),
            sc * (cA * sC + cC * sA * sB) + cc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa),
            cb * (sa * (cA * cC - sA * sB * sC) - cB * sA * ca) - sb * (
                        cc * (cA * sC + cC * sA * sB) - sc * (ca * (cA * cC - sA * sB * sC) + cB * sA * sa)),
        ],
        [
            sb * (sa * (cC * sA + cA * sB * sC) + cA * cB * ca) + cb * (
                        cc * (sA * sC - cA * cC * sB) - sc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa)),
            cc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa) + sc * (sA * sC - cA * cC * sB),
            cb * (sa * (cC * sA + cA * sB * sC) + cA * cB * ca) - sb * (
                        cc * (sA * sC - cA * cC * sB) - sc * (ca * (cC * sA + cA * sB * sC) - cA * cB * sa)),
        ],
    ]), np.array([
        [- Tx - z * (sB * ca - cB * sC * sa)],
        [- Ty - z * (sa * (cA * cC - sA * sB * sC) - cB * sA * ca)],
        [- Tz - z * (sa * (cC * sA + cA * sB * sC) + cA * cB * ca)]
    ])


class PinBallsDisplayWidget(QWidget):
    def __init__(self, udp_receiver):
        super().__init__()
        self.udp_receiver = udp_receiver

        # 电机限位
        self.a_threshold = [2000, 4000]
        self.b_threshold = [2100, 3000]

        self.init_ui()

        # 连接数据更新信号
        self.udp_receiver.data_refreshed.connect(self.update_display)

        # 串口初始化
        self.serial = QSerialPort(self)
        self.serial.setPortName("COM3")
        self.serial.setBaudRate(115200)
        self.serial.setDataBits(QSerialPort.Data8)
        self.serial.setParity(QSerialPort.NoParity)
        self.serial.setStopBits(QSerialPort.OneStop)

        self.connect_serial()

    def init_ui(self):
        self.setWindowTitle("Pin Balls 实时坐标显示与电机控制")
        self.setGeometry(100, 100, 600, 500)

        layout = QVBoxLayout()

        # ===== 标题 =====
        title_label = QLabel("Pin Balls 实时监控与电机控制")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)

        # ===== Pin Balls 显示 =====
        self.pin_label = QLabel("等待数据...")
        self.pin_label.setStyleSheet("font-size: 18px; color: green;")
        self.pin_label.setWordWrap(True)
        layout.addWidget(self.pin_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # ===== 电机位置显示 =====
        self.arm_pos_label = QLabel("电机位置：等待响应...")
        self.arm_pos_label.setStyleSheet("font-size: 18px; color: blue;")
        layout.addWidget(self.arm_pos_label)

        self.get_current_position_btn = QPushButton("获取当前位置")
        self.get_current_position_btn.clicked.connect(self.get_current_position)
        layout.addWidget(self.get_current_position_btn)

        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)

        # ===== 电机 A 控制 =====
        layout_a = QHBoxLayout()
        label_a = QLabel("电机 A:")
        label_a.setStyleSheet("font-size: 18px;")
        self.a_value_label = QLabel(str(self.a_threshold[0]))
        self.a_value_label.setStyleSheet("font-size: 18px; min-width: 80px;")

        self.a_slider = QSlider(Qt.Horizontal)
        self.a_slider.setMinimum(self.a_threshold[0])
        self.a_slider.setMaximum(self.a_threshold[1])
        self.a_slider.setValue(self.a_threshold[0])
        self.a_slider.valueChanged.connect(self.on_a_slider_changed)

        layout_a.addWidget(label_a)
        layout_a.addWidget(self.a_slider)
        layout_a.addWidget(self.a_value_label)
        layout.addLayout(layout_a)

        # ===== 电机 B 控制 =====
        layout_b = QHBoxLayout()
        label_b = QLabel("电机 B:")
        label_b.setStyleSheet("font-size: 18px;")
        self.b_value_label = QLabel(str(self.b_threshold[0]))
        self.b_value_label.setStyleSheet("font-size: 18px; min-width: 80px;")

        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.setMinimum(self.b_threshold[0])
        self.b_slider.setMaximum(self.b_threshold[1])
        self.b_slider.setValue(self.b_threshold[0])
        self.b_slider.valueChanged.connect(self.on_b_slider_changed)

        layout_b.addWidget(label_b)
        layout_b.addWidget(self.b_slider)
        layout_b.addWidget(self.b_value_label)
        layout.addLayout(layout_b)

        # ===== 应用按钮（可选）=====
        self.move_btn = QPushButton("立即移动电机")
        self.move_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.move_btn.clicked.connect(self.send_move_command)
        layout.addWidget(self.move_btn)

        # 添加位置反解
        self.position_reverse_label = QLabel("位置反解：")
        self.position_reverse_label.setStyleSheet("font-size: 18px; color: blue;")
        layout.addWidget(self.position_reverse_label)

        self.setLayout(layout)

    def connect_serial(self):
        if self.serial.isOpen():
            print("串口已打开")
            return
        if self.serial.open(QSerialPort.ReadWrite):
            print(f"✅ 成功连接到串口 {self.serial.portName()}")
        else:
            error = self.serial.errorString()
            print(f"❌ 无法打开串口: {error}")

    def read_serial_data(self):
        if self.serial.waitForReadyRead(1000):
            data = self.serial.readAll().data().decode().strip()
            return data
        else:
            return None

    def get_current_position(self):
        command = "GET\n"
        self.serial.write(command.encode())
        data = self.read_serial_data()
        if data:
            lines = data.splitlines()
            if len(lines) >= 2:
                a_pos = lines[0].strip()
                b_pos = lines[1].strip()
                self.arm_pos_label.setText(f"电机位置：A={a_pos}, B={b_pos}")
                a_pos = (float(a_pos)/2048)*np.pi
                b_pos = (float(b_pos)/2048)*np.pi
                Rend, Tend = compute_Tend(float(a_pos), float(b_pos))
                arm_pos_0 = Rend @ np.array(param_toolP[0]).reshape(3, 1) + Tend
                arm_pos_1 = Rend @ np.array(param_toolP[1]).reshape(3, 1) + Tend
                self.position_reverse_label.setText(f"位置反解：{arm_pos_0.ravel()}，{arm_pos_1.ravel()}")
                # arm_pos = arm_pos_to_world(a_pos, b_pos)
                # self.position_reverse_label.setText(f"位置反解：{arm_pos}")
            else:
                self.arm_pos_label.setText("电机位置：解析失败")
        else:
            self.arm_pos_label.setText("电机位置：超时或无响应")

    @pyqtSlot()
    def update_display(self):
        pin_data = self.udp_receiver.pin_balls_in_cam
        if pin_data is not None and len(pin_data) == 2:
            p1, p2 = pin_data[0], pin_data[1]
            text = (
                "Pin Balls 坐标:\n"
                f"  P1: ({p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f})\n"
                f"  P2: ({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f})"
            )
            self.pin_label.setText(text)
        else:
            self.pin_label.setText("Pin Balls: 数据未就绪")

    def on_a_slider_changed(self, value):
        self.a_value_label.setText(str(value))

    def on_b_slider_changed(self, value):
        self.b_value_label.setText(str(value))

    def send_move_command(self):
        a = self.a_slider.value()
        b = self.b_slider.value()
        command = f"MOVE {a} {b} 300 300\n"
        if self.serial.isOpen():
            self.serial.write(command.encode())
            print(f"发送命令: {command.strip()}")
        else:
            print("串口未打开，无法发送命令")

    def closeEvent(self, event):
        self.udp_receiver.stop()
        if self.serial.isOpen():
            self.serial.close()
            print("串口已关闭")
        event.accept()




# ===== 主程序入口 =====
if __name__ == "__main__":
    udp_receiver = UdpReceiverThread(ip="127.0.0.1", port=8080, n_average=10)
    udp_receiver.start()

    app = QApplication(sys.argv)
    window = PinBallsDisplayWidget(udp_receiver)
    window.show()

    sys.exit(app.exec_())