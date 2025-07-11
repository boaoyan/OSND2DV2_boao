from ui_interaction.ui_build.init_para import InitPara


class EventHandler(InitPara):
    def __init__(self):
        super().__init__()

    def showEvent(self, a0):
        self.view_manager.resize_event()

    def resizeEvent(self, a0):
        self.view_manager.resize_event()