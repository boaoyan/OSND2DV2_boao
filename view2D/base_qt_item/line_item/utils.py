from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGraphicsLineItem


def draw_line(color, thickness, style):
    line = QGraphicsLineItem(0, 0, 1, 1)
    pen = QPen(color, thickness, style)
    line.setPen(pen)
    line.setVisible(False)
    line.setZValue(1)
    return line