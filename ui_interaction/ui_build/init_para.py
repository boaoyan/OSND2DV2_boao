from ui_interaction.ui_build.init_layout import InitUILayout
from view2D.view_manager import ViewerManager
from view2D.view_render import ViewRender
from config import ConfigManager
import cv2 as cv


class InitPara(InitUILayout):
    def __init__(self):
        super().__init__()
        # 导入数据配置文件
        self.data_config = ConfigManager().get_instance().get_all_configs("default_data_path_config")
        # 初始化视图显示窗口
        front_img = cv.imread(self.data_config["ct_image"]["front_img_path"], 0)
        self.front_view_render = ViewRender(self.front_view, front_img)
        side_img = cv.imread(self.data_config["ct_image"]["side_img_path"], 0)
        self.side_view_render = ViewRender(self.side_view, side_img)
        self.view_manager = ViewerManager()
        self.view_manager.viewers.append(self.front_view_render)
        self.view_manager.viewers.append(self.side_view_render)
