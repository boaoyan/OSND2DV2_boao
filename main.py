import sys
import toml
from PyQt5.QtWidgets import QApplication

from camera_communication.wave_view.time_series_viewer import DualPointVisualizer
from config import ConfigManager
from ui_interaction import EventHandler


class QRobotControlWidget(EventHandler):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    # 创建 ConfigManager 单例实例
    config_manager = ConfigManager.get_instance()
    with open('config/all_config_path.toml', 'r', encoding='utf-8') as f:
        config_files = toml.load(f)
        config_manager.load_configs(config_files['config_files'])
        app = QApplication(sys.argv)

        # ✅ 1. 创建主窗口
        pinWidget = QRobotControlWidget()
        pinWidget.resize(900, 600)
        pinWidget.show()

        # # ✅ 2. 创建独立可视化窗口（parent=None）
        # viz_window = DualPointVisualizer(parent=None, history_length=200)
        #
        # viz_window.setWindowTitle("📊 实时波形监控（独立窗口）")
        # viz_window.show()

        # # ✅ 3. 【关键】将可视化窗口传递给主窗口，建立信号连接
        # pinWidget.set_visualizer(viz_window)
        #
        # # ✅ 4. 保存引用（便于关闭时管理）
        # pinWidget.viz_window = viz_window
        #
        # print("✅ 双窗口模式启动：主控制窗口 + 独立可视化窗口")

        sys.exit(app.exec())
