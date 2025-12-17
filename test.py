import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

app = QApplication(sys.argv)
window = QMainWindow()
central_widget = QWidget()
layout = QVBoxLayout(central_widget)
window.setCentralWidget(central_widget)

# 关键：测试 antialiasing=True
plotter = QtInteractor(central_widget)
plotter.ren_win.SetMultiSamples(4)
layout.addWidget(plotter)

window.show()
sys.exit(app.exec_())