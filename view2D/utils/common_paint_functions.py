"""
记录所有渲染器的公共绘制函数
"""
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtWidgets import QGraphicsEllipseItem


def draw_circle(circle_para, diameter, is_filled, is_visible=False):
    # 创建一个 QGraphicsEllipseItem 对象（圆）
    radius = diameter / 2
    # 定义圆形状，位置稍后设置
    rect = QRectF(0, 0, diameter, diameter)
    circle = QGraphicsEllipseItem(rect)
    circle.setPos(circle_para["pos"][0] - radius,
                  circle_para["pos"][1] - radius)  # 设置圆的位置，使其中心位于钉子的旋转中心
    # 创建一个 QPen 对象，并设置线宽
    pen = QPen(circle_para["color"], circle_para["thickness"])  # 红色，线宽为 1
    if is_filled:
        # 设置圆的填充颜色为红色
        circle.setBrush(QBrush(circle_para['fill_color']))
    circle.setPen(pen)  # 将 QPen 对象应用到 QGraphicsEllipseItem 上
    if is_visible:
        circle.setVisible(True)
    else:
        circle.setVisible(False)
    circle.setZValue(1)
    return circle

def set_circle_color(circle, thickness, new_color, is_filled):
    if new_color is not None:
        pen = QPen(new_color, thickness)
        circle.setPen(pen)
        if is_filled:
            circle.setBrush(QBrush(new_color))

def set_line_color(line, thickness, new_color, style):
    if new_color is not None:
        pen = QPen(new_color, thickness, style)
        line.setPen(pen)

def set_visible_rect(vision_render, x, y, shape):
    width, height = shape
    vision_render.visible_rect_para["x"] = x
    vision_render.visible_rect_para["y"] = y
    vision_render.visible_rect_para["width"] = width
    vision_render.visible_rect_para["height"] = height

def show_visible_rect(vision_render):
    x = vision_render.visible_rect_para["x"]
    y = vision_render.visible_rect_para["y"]
    width = vision_render.visible_rect_para["width"]
    height = vision_render.visible_rect_para["height"]
    if width is not None and height is not None:
        vision_render.visible_rect.setRect(QRectF(x, y, width, height))
        vision_render.visible_rect.setVisible(True)