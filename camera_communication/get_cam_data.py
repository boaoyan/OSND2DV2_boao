import json
import socket
import numpy as np
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal


class UdpReceiverThread(QThread):
    data_received = pyqtSignal(str)
    data_refreshed = pyqtSignal()  # 当两个数据都更新并平均后触发

    def __init__(self, ip="127.0.0.1", port=8080, n_average=10):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = True
        self.should_listen = False
        self.n_average = max(1, n_average)

        # 当前平均后的结果
        self.ct_balls_in_cam = None
        self.pin_order_flipped = False  # False: 正常顺序；True: 交换顺序
        self._pin_balls_in_cam = None
        # 新增：存储上一次的有效平均值，用于变化检测
        self._last_ct_avg = None  # shape (4,)
        self._last_pin_avg = None  # shape (2, 3)
        # 缓冲区：使用 deque 自动维护长度
        self.ct_buffer = deque(maxlen=self.n_average)
        self.pin_buffer = deque(maxlen=self.n_average)

        self.data_received.connect(self.update_data)

    @property
    def pin_balls_in_cam(self):
        if self._pin_balls_in_cam is None:
            return None
        return self._pin_balls_in_cam if not self.pin_order_flipped else self._pin_balls_in_cam[::-1]

    def run(self):
        print("UDP receiver thread started, waiting for listen trigger...")
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind((self.ip, self.port))
        udp_socket.settimeout(1)

        while self.running:
            if not self.should_listen:
                # 尚未触发监听，短暂休眠后继续检查
                self.msleep(50)  # QThread 的 sleep，非 time.sleep
                continue

            try:
                data, addr = udp_socket.recvfrom(1024)
                message = data.decode("utf-8")
                self.data_received.emit(message)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {e}")
                break

        udp_socket.close()

    def start_listening(self):
        """外部调用此函数，开始接收 UDP 数据"""
        self.should_listen = True


    def stop(self):
        self.running = False
        self.wait()

    def align_pin_points(self, p1, p2):
        """
        根据当前 pin_balls_in_cam[0] 的位置，自动对齐两个点的顺序。
        使得第一个点更接近当前的参考点（self._pin_balls_in_cam[0]）。

        参数:
            p1, p2: list 或 tuple, 三维点 [x, y, z]

        返回:
            list: 对齐后的 [point1, point2]，顺序已调整
        """
        if self._pin_balls_in_cam is None:
            return [p1, p2]  # 未初始化，不调整

        try:
            point1 = np.array(p1)
            point2 = np.array(p2)
            reference = np.array(self._pin_balls_in_cam[0])  # 当前第一个点作为参考

            dist1 = np.linalg.norm(point1 - reference)
            dist2 = np.linalg.norm(point2 - reference)

            if dist2 < dist1:
                # print(f"自动对齐：交换点顺序以匹配当前状态")
                return [p2, p1]
            else:
                return [p1, p2]
        except Exception as e:
            print(f"对齐点时出错: {e}")
            return [p1, p2]

    def update_data(self, data):
        try:
            data = json.loads(data)
            # print(data)
        except json.JSONDecodeError:
            print("Invalid JSON received")
            return

        if "template_uint8_t" not in data or "points_poi" not in data:
            return

        template_id = data["template_uint8_t"]
        points_poi = data["points_poi"]

        updated = False

        if template_id == 1 and len(points_poi) > 0:
            ct_data = points_poi[0]
            if len(ct_data) == 4:
                self.ct_buffer.append(np.array(ct_data))
                updated = True

        elif template_id == 4 and len(points_poi) > 0:
            pin_data = points_poi[0]
            if len(pin_data) == 2:
                pin_data = self.align_pin_points(pin_data[0], pin_data[1])
            self.pin_buffer.append(np.array(pin_data))
            updated = True

        elif template_id == 5 and len(points_poi) > 1:
            ct_data = points_poi[0]
            pin_data = points_poi[1]
            if len(ct_data) == 4:
                self.ct_buffer.append(np.array(ct_data))
                updated = True
            if len(pin_data) == 2:
                pin_data = self.align_pin_points(pin_data[0], pin_data[1])
                self.pin_buffer.append(np.array(pin_data))
                updated = True

        # --- 新增：变化检测逻辑 ---
        need_refresh = False
        # 处理 CT 球
        if len(self.ct_buffer) >= 1:
            new_ct_avg = np.mean(self.ct_buffer, axis=0)  # shape (4,)
            if self._last_ct_avg is None:
                # 首次更新，直接接受
                self.ct_balls_in_cam = new_ct_avg
                self._last_ct_avg = new_ct_avg.copy()
                need_refresh = True
            else:
                # 仅比较前3维（位置）
                dist = np.linalg.norm(new_ct_avg[:3] - self._last_ct_avg[:3])
                if dist > 0.2:
                    self.ct_balls_in_cam = new_ct_avg
                    self._last_ct_avg = new_ct_avg.copy()
                    need_refresh = True

        # 当缓冲区有数据时，触发刷新
        if len(self.pin_buffer) >= 1:
            new_pin_avg = np.mean(self.pin_buffer, axis=0)  # shape (2, 3)
            if self._last_pin_avg is None:
                self._pin_balls_in_cam = new_pin_avg
                self._last_pin_avg = new_pin_avg.copy()
                need_refresh = True
            else:
                # 计算整体欧氏距离（Frobenius 范数）
                dist = np.linalg.norm(new_pin_avg - self._last_pin_avg)
                if dist > 0.2:
                    self._pin_balls_in_cam = new_pin_avg
                    self._last_pin_avg = new_pin_avg.copy()
                    need_refresh = True
                # else: 不更新

        if need_refresh:
            self.data_refreshed.emit()


    def set_n_average(self, n):
        """动态设置平均次数 N"""
        n = max(1, n)
        if n != self.n_average:
            self.n_average = n
            # 重建缓冲区，保留现有数据但限制新长度
            if self.ct_buffer:
                new_ct = deque(self.ct_buffer, maxlen=n)
                self.ct_buffer = new_ct
            else:
                self.ct_buffer = deque(maxlen=n)

            if self.pin_buffer:
                new_pin = deque(self.pin_buffer, maxlen=n)
                self.pin_buffer = new_pin
            else:
                self.pin_buffer = deque(maxlen=n)

    def get_ct_buffer_size(self):
        return len(self.ct_buffer)

    def get_pin_buffer_size(self):
        return len(self.pin_buffer)

    def toggle_pin_pos_order(self):
        self.pin_order_flipped = not self.pin_order_flipped