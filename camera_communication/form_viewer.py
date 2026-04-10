# 在主窗口类中（或单独管理器类）
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.axs = self.fig.subplots(3, 2, sharex=True)  # 3行2列：X1/X2, Y1/Y2, Z1/Z2
        self.fig.tight_layout()

    def update_plot(self, raw_hist, filtered_hist, pin_order_flipped=False):
        """
        raw_hist: list of [x1,y1,z1,x2,y2,z2]
        filtered_hist: same shape
        """
        if not raw_hist or not filtered_hist:
            return

        # 转为 (N, 6) 数组
        raw = np.array(raw_hist)          # shape (N, 6)
        filt = np.array(filtered_hist)

        # 如果需要翻转点顺序，则交换前3和后3列
        if pin_order_flipped:
            raw = np.concatenate([raw[:, 3:6], raw[:, 0:3]], axis=1)
            filt = np.concatenate([filt[:, 3:6], filt[:, 0:3]], axis=1)

        N = raw.shape[0]
        t = np.arange(N)

        labels = ['X₁', 'X₂', 'Y₁', 'Y₂', 'Z₁', 'Z₂']
        colors = ['red', 'green']

        # 清除
        for ax in self.axs.flat:
            ax.clear()

        # 绘制
        axes = self.axs
        # 第一列：点1 (X1, Y1, Z1)
        axes[0, 0].plot(t, raw[:, 0], '--', color='red', label='Raw X₁')
        axes[0, 0].plot(t, filt[:, 0], '-', color='green', label='Filtered X₁')
        axes[1, 0].plot(t, raw[:, 1], '--', color='red', label='Raw Y₁')
        axes[1, 0].plot(t, filt[:, 1], '-', color='green', label='Filtered Y₁')
        axes[2, 0].plot(t, raw[:, 2], '--', color='red', label='Raw Z₁')
        axes[2, 0].plot(t, filt[:, 2], '-', color='green', label='Filtered Z₁')

        # 第二列：点2 (X2, Y2, Z2)
        axes[0, 1].plot(t, raw[:, 3], '--', color='red', label='Raw X₂')
        axes[0, 1].plot(t, filt[:, 3], '-', color='green', label='Filtered X₂')
        axes[1, 1].plot(t, raw[:, 4], '--', color='red', label='Raw Y₂')
        axes[1, 1].plot(t, filt[:, 4], '-', color='green', label='Filtered Y₂')
        axes[2, 1].plot(t, raw[:, 5], '--', color='red', label='Raw Z₂')
        axes[2, 1].plot(t, filt[:, 5], '-', color='green', label='Filtered Z₂')

        for i, ax in enumerate(axes.flat):
            ax.legend(loc='upper right')
            ax.grid(True)
            if i >= 4:  # 最后一行
                ax.set_xlabel('Time (frame)')

        self.canvas.draw()