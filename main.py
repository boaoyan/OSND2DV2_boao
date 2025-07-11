import sys
import toml
from PyQt5.QtWidgets import QApplication

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
        pinWidget = QRobotControlWidget()
        # 设置窗口的初始大小
        pinWidget.show()
        sys.exit(app.exec())
