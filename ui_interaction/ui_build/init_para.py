from ui_interaction.ui_build.init_layout import InitUILayout
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
        guide_event_params = {
            "view_manager": self.view_manager,
            "sz_view_render": self.front_view_render,
            "sc_view_render": self.side_view_render,
            "start_gui_btn": self.start_gui_btn,
            "finish_gui_btn": self.finish_gui_btn,
            "cancel_gui_btn": self.cancel_gui_btn,
            "a_arm": self.data_config["trans_matrix"]["a_arm"],
            "ct_pos_label": self.ct_pos_label
        }
        self.guide_event = GuideEvent(**guide_event_params)
