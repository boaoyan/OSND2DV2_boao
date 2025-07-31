import sys

from PyQt5.QtWidgets import QMainWindow, QApplication

from camera_communication.get_cam_data import UdpReceiverThread


class TestCam(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Test Camera')
        self.setGeometry(100, 100, 800, 600)
        # 启动相机线程
        self.camera_thread = UdpReceiverThread()
        self.camera_thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestCam()
    window.show()
    sys.exit(app.exec_())