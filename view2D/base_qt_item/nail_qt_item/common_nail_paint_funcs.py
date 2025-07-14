"""
这里放置的是nail_qt_item_x/y/z通用的一些钉子图元绘制函数
"""
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem


def draw_nail_rect(hat_pos, hat_size, hat_style, pin_pos, pin_size, pin_style):
    """
    绘制侧面的钉子，两个矩形
    钉身的矩形左边中点为针尖所在的位置，钉帽的矩形
    :return:
    """
    # section 1 绘制钉帽
    hat = QRectF(hat_pos[0], hat_pos[1], hat_size[0], hat_size[1])
    hat_rect = QGraphicsRectItem(hat)
    # 创建一个 QPen 对象，并设置线宽
    pen = QPen(QColor(hat_style["color"]), hat_style["thickness"])  # 红色，线宽为 2
    hat_rect.setPen(pen)  # 将 QPen 对象应用到 QGraphicsEllipseItem 上
    # section 2 绘制钉身
    pin = QRectF(pin_pos[0], pin_pos[1], pin_size[0], pin_size[1])
    pin_rect = QGraphicsRectItem(pin)
    # 创建一个 QPen 对象，并设置线宽
    border_color = QColor(pin_style["color"])
    border_color.setAlpha(128)
    pen = QPen(border_color)  # 绿色
    pen.setWidth(pin_style["thickness"])  # 线宽为 1
    pen.setStyle(pin_style["line_style"])
    pin_rect.setPen(pen)  # 将 QPen 对象应用到 QGraphicsRectItem 上
    return hat_rect, pin_rect

def draw_nail_circle(outer_diameter, inner_diameter, pos, hat_style, pin_style):
    # 创建一个 QGraphicsEllipseItem 对象（圆）
    radius = outer_diameter / 2
    # 定义圆形状，位置稍后设置
    rect = QRectF(0, 0, outer_diameter, outer_diameter)
    hat_circle = QGraphicsEllipseItem(rect)
    hat_circle.setPos(pos[1] - radius, pos[0] - radius)  # 设置圆的位置，使其中心位于钉子的旋转中心
    # 创建一个 QPen 对象，并设置线宽
    pen = QPen(QColor(hat_style["color"]), hat_style["thickness"])  # 红色，线宽为 2
    hat_circle.setPen(pen)  # 将 QPen 对象应用到 QGraphicsEllipseItem 上
    # 添加中间的小圆
    radius = inner_diameter / 2
    rect = QRectF(0, 0, inner_diameter, inner_diameter)
    pin_circle = QGraphicsEllipseItem(rect)
    pin_circle.setPos(pos[1] - radius, pos[0] - radius)
    # 创建一个 QPen 对象，并设置线宽
    border_color = QColor(pin_style["color"])
    border_color.setAlpha(128)
    pen = QPen(border_color)  # 绿色
    pen.setWidth(pin_style["thickness"])  # 线宽为 1
    pen.setStyle(pin_style["line_style"])
    pin_circle.setPen(pen)  # 将 QPen 对象应用到 QGraphicsEllipseItem 上
    return hat_circle, pin_circle