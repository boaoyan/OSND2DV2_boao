import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout

app = QApplication(sys.argv)

window = QWidget()
layout = QVBoxLayout()

graphics_view = QGraphicsView()
scene = QGraphicsScene()
scene.addText("Hello QGraphicsView")
graphics_view.setScene(scene)

# 设置边框样式
graphics_view.setStyleSheet("border: 3px solid blue;")

layout.addWidget(graphics_view)
window.setLayout(layout)
window.resize(400, 300)
window.show()

sys.exit(app.exec_())