from PyQt5.QtWidgets import QMessageBox


def messageBox(text: str):
    # 创建一个QMessageBox实例
    msgBox = QMessageBox()

    # 设置消息框的图标为警告
    msgBox.setIcon(QMessageBox.Icon.Warning)

    # 设置消息框的标题和文本
    msgBox.setWindowTitle("警告")
    msgBox.setText(text)

    # 设置标准按钮（这里我们使用Ok和Cancel）
    msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)

    # 显示消息框并获取用户点击的按钮
    # exec()方法会阻塞，直到用户关闭了消息框
    # 它返回用户点击的按钮的QMessageBox.StandardButton枚举值
    result = msgBox.exec()

    return result