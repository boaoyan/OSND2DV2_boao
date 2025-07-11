class ViewerManager:
    def __init__(self):
        self.viewers = []

    def resize_event(self):
        for viewer in self.viewers:
            viewer.resize_event()