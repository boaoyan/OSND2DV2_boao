from view2D.view_render import ViewRender


class ViewerManager:
    def __init__(self):
        self.viewers = []
        self.active_view = None

    def resize_event(self):
        for viewer in self.viewers:
            viewer.resize_event()

    def update_activated_view(self, viewer):
        for v in self.viewers:
            if v is viewer:
                assert isinstance(v, ViewRender)
                v.activate_self()
                v.update_uv()
                self.active_view = v
            else:
                v.deactivate_self()

    def deactivate_all(self):
        self.active_view = None
        for v in self.viewers:
            v.deactivate_self()
