# import numpy as np
#
# from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel
# from PyQt5.QtCore import QTimer, Qt
# import pyqtgraph as pg  # 需要安装：pip install pyqtgraph
#
#
# class DualPointVisualizer(QWidget):
#     """双点刚性连接实时可视化器（2D 时间序列）"""
#
#     def __init__(self, parent=None, history_length=200, update_rate=30):
#         super().__init__(parent)
#         self.history_length = history_length
#
#         # 窗口设置（独立窗口时）
#         if parent is None:
#             self.setAttribute(Qt.WA_DeleteOnClose)
#             self.setWindowTitle("实时波形监控")
#
#         self.history_length = history_length
#         self.is_updating = True
#         self.frame_count = 0
#
#         # ✅ 主布局
#         main_layout = QVBoxLayout(self)
#         main_layout.setContentsMargins(5, 5, 5, 5)
#         main_layout.setSpacing(5)
#
#         # 标题
#         title_label = QLabel("📊 双点坐标时间序列（测量 vs 滤波）")
#         title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
#         main_layout.addWidget(title_label)
#
#         # 创建 6 个子图（3 轴 × 2 点）
#         self.plots = {}
#         self.curves = {}
#         self.data_buffers = {}
#
#         grid_layout = QGridLayout()
#
#         # 坐标轴标签
#         axes = ['X', 'Y', 'Z']
#         points = ['A', 'B']
#         colors = {
#             'meas': {'A': '#FF6B6B', 'B': '#FF9999'},  # 测量值 - 红色系
#             'filt': {'A': '#4ECDC4', 'B': '#99E6E0'}  # 滤波值 - 青色系
#         }
#
#         for i, axis in enumerate(axes):
#             for j, point in enumerate(points):
#                 # 创建图表
#                 plot_widget = pg.PlotWidget()
#                 plot_widget.setTitle(f"{point}点 - {axis}轴")
#                 plot_widget.setLabel('left', f'{axis} (mm)')
#                 plot_widget.setLabel('bottom', '帧')
#                 plot_widget.showGrid(x=True, y=True, alpha=0.3)
#                 plot_widget.setMaximumHeight(150)
#
#                 # 初始化数据缓冲区
#                 key_meas = f"{point}_{axis}_meas"
#                 key_filt = f"{point}_{axis}_filt"
#                 self.data_buffers[key_meas] = np.zeros(history_length)
#                 self.data_buffers[key_filt] = np.zeros(history_length)
#
#                 # 创建曲线
#                 self.curves[key_meas] = plot_widget.plot(
#                     pen=pg.mkPen(color=colors['meas'][point], width=1),
#                     name='Measurement'
#                 )
#                 self.curves[key_filt] = plot_widget.plot(
#                     pen=pg.mkPen(color=colors['filt'][point], width=2, style=Qt.DashLine),
#                     name='Filtered'
#                 )
#
#                 # 添加图例
#                 plot_widget.addLegend()
#
#                 # 限制 Y 轴范围（可选，根据实际数据调整）
#                 # plot_widget.setYRange(-200, 700)
#
#                 grid_layout.addWidget(plot_widget, i, j)
#                 self.plots[f"{point}_{axis}"] = plot_widget
#
#         main_layout.addLayout(grid_layout)
#
#         # 控制按钮
#         btn_layout = QVBoxLayout()
#         self.btn_toggle = QPushButton("⏸️ 暂停/继续")
#         self.btn_toggle.clicked.connect(self.toggle_update)
#         self.btn_reset = QPushButton("🔄 重置视图")
#         self.btn_reset.clicked.connect(self.reset_view)
#         btn_layout.addWidget(self.btn_toggle)
#         btn_layout.addWidget(self.btn_reset)
#         main_layout.addLayout(btn_layout)
#
#         # ✅ 使用 QTimer 精确控制更新频率
#         self.update_rate = update_rate
#         self.pending_data = None  # 缓存最新数据
#
#         self.viz_timer = QTimer()
#         self.viz_timer.timeout.connect(self._render_pending_data)
#         self.viz_timer.start(1000 // update_rate)  # 精确 30Hz
#
#         print(f"✅ 可视化定时器启动：{update_rate}Hz")
#     def update_data(self, measurement, filtered_state):
#         """接收数据并缓存（不立即绘图）"""
#         if not self.is_updating:
#             return
#
#         # 只缓存最新数据，避免队列堆积
#         self.pending_data = (measurement, filtered_state)
#
#     def _render_pending_data(self):
#         """定时器触发时执行绘图"""
#         if not self.pending_data or not self.is_updating:
#             return
#
#         measurement, filtered_state = self.pending_data
#         self.pending_data = None  # 消费数据
#
#         meas_A, meas_B = measurement[0], measurement[1]  # 每个 shape (3,)
#         filt_A, filt_B = filtered_state[0], filtered_state[1]
#
#         # 坐标轴映射
#         axes = ['X', 'Y', 'Z']
#         points_data = {
#             'A': {'meas': meas_A, 'filt': filt_A},
#             'B': {'meas': meas_B, 'filt': filt_B}
#         }
#
#         # 更新每个坐标轴的数据
#         for i, axis in enumerate(axes):
#             for point, data in points_data.items():
#                 key_meas = f"{point}_{axis}_meas"
#                 key_filt = f"{point}_{axis}_filt"
#
#                 # 滚动缓冲区（移除最旧数据，添加新数据）
#                 self.data_buffers[key_meas] = np.roll(self.data_buffers[key_meas], -1)
#                 self.data_buffers[key_filt] = np.roll(self.data_buffers[key_filt], -1)
#                 self.data_buffers[key_meas][-1] = data['meas'][i]
#                 self.data_buffers[key_filt][-1] = data['filt'][i]
#
#                 # 更新曲线
#                 x_data = np.arange(self.frame_count - self.history_length + 1, self.frame_count + 1)
#                 self.curves[key_meas].setData(x_data, self.data_buffers[key_meas])
#                 self.curves[key_filt].setData(x_data, self.data_buffers[key_filt])
#
#         self.frame_count += 1
#
#     def toggle_update(self):
#         """暂停/继续更新"""
#         self.is_updating = not self.is_updating
#         if self.is_updating:
#             self.btn_toggle.setText("⏸️ 暂停/继续")
#         else:
#             self.btn_toggle.setText("▶️ 暂停/继续")
#
#     def reset_view(self):
#         """重置所有数据缓冲区"""
#         for key in self.data_buffers:
#             self.data_buffers[key] = np.zeros(self.history_length)
#         self.frame_count = 0
#         for curve in self.curves.values():
#             curve.setData([], [])


import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel,
    QTabWidget, QScrollArea, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
import pyqtgraph as pg


class DualPointVisualizer(QWidget):
    """双模块实时可视化器：独立位置滤波 + 关联位置滤波"""

    def __init__(self, parent=None, history_length=200, update_rate=30):
        super().__init__(parent)

        # 窗口设置（独立窗口时）
        if parent is None:
            self.setAttribute(Qt.WA_DeleteOnClose)
            self.setWindowTitle("📊 双模块波形监控")
            self.resize(1600, 1000)

        self.history_length = history_length
        self.is_updating = True
        self.frame_count = 0

        # ✅ 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # ✅ 标题
        title_label = QLabel("📊 双模块坐标时间序列对比")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # ✅ 创建标签页容器
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setMovable(False)
        main_layout.addWidget(self.tab_widget)

        # ✅ 初始化两个模块
        self._init_module1_independent()  # 模块 1：独立位置滤波
        self._init_module2_correlated()  # 模块 2：关联位置滤波

        # ✅ 控制按钮（全局）
        btn_layout = QVBoxLayout()
        self.btn_toggle = QPushButton("⏸️ 暂停/继续")
        self.btn_toggle.clicked.connect(self.toggle_update)
        self.btn_toggle.setFixedHeight(35)

        self.btn_reset = QPushButton("🔄 重置所有视图")
        self.btn_reset.clicked.connect(self.reset_all_views)
        self.btn_reset.setFixedHeight(35)

        btn_layout.addWidget(self.btn_toggle)
        btn_layout.addWidget(self.btn_reset)
        main_layout.addLayout(btn_layout)

        # ✅ 定时器控制更新频率
        self.update_rate = update_rate
        self.pending_data = {'module1': None, 'module2': None}

        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self._render_pending_data)
        if update_rate > 0:
            self.viz_timer.start(1000 // update_rate)
            print(f"✅ 可视化定时器启动：{update_rate}Hz")

        print("✅ DualPointVisualizer 双模块初始化完成")

    # ─────────────────────────────────────────────────────────────
    # 模块 1：独立位置滤波（保持原状：6 子图，2 点×3 轴）
    # ─────────────────────────────────────────────────────────────
    def _init_module1_independent(self):
        """初始化模块 1：独立位置滤波（A/B 两点，每点 3 轴）"""
        module1_widget = QWidget()
        layout = QVBoxLayout(module1_widget)

        # 模块标题
        title = QLabel("🔹 模块 1：独立位置滤波（双点刚性约束）")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E86AB;")
        layout.addWidget(title)

        # 配置
        axes = ['X', 'Y', 'Z']
        points = ['A', 'B']
        colors = {
            'meas': {'A': '#FF6B6B', 'B': '#FF9999'},
            'filt': {'A': '#4ECDC4', 'B': '#99E6E0'}
        }

        # 创建 6 个子图
        self.m1_plots = {}
        self.m1_curves = {}
        self.m1_buffers = {}

        grid = QGridLayout()
        grid.setSpacing(5)

        for i, axis in enumerate(axes):
            for j, point in enumerate(points):
                plot = pg.PlotWidget()
                plot.setTitle(f"{point}点 - {axis}轴")
                plot.setLabel('left', f'{axis} (mm)')
                plot.setLabel('bottom', '帧')
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setBackground('w')
                plot.setMinimumHeight(100)
                plot.setMaximumHeight(140)

                key_meas = f"{point}_{axis}_meas"
                key_filt = f"{point}_{axis}_filt"

                self.m1_buffers[key_meas] = np.zeros(self.history_length)
                self.m1_buffers[key_filt] = np.zeros(self.history_length)

                self.m1_curves[key_meas] = plot.plot(
                    pen=pg.mkPen(color=colors['meas'][point], width=1),
                    name='Measurement'
                )
                self.m1_curves[key_filt] = plot.plot(
                    pen=pg.mkPen(color=colors['filt'][point], width=2, style=Qt.DashLine),
                    name='Filtered'
                )

                plot.addLegend()
                grid.addWidget(plot, i, j)
                self.m1_plots[f"{point}_{axis}"] = plot

        # 使用 ScrollArea 防止窗口过高
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(QWidget())
        scroll.widget().setLayout(grid)
        layout.addWidget(scroll)

        # 添加标签页
        self.tab_widget.addTab(module1_widget, "🔹 独立位置滤波")

    # ─────────────────────────────────────────────────────────────
    # 模块 2：关联位置滤波（7 维观测值）
    # ─────────────────────────────────────────────────────────────
    def _init_module2_correlated(self):
        """初始化模块 2：关联位置滤波（7 维观测值）"""
        module2_widget = QWidget()
        layout = QVBoxLayout(module2_widget)

        # 模块标题 + 说明
        header = QWidget()
        header_layout = QVBoxLayout(header)

        title = QLabel("🔸 模块 2：关联位置滤波（7 维联合观测）")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #A23B72;")

        desc = QLabel("观测值格式: [x₁, y₁, z₁, w, x₂, y₂, z₂]\n"
                      "• (x₁,y₁,z₁): 点 1 位置  • w: 关联权重/四元数分量  • (x₂,y₂,z₂): 点 2 位置")
        desc.setStyleSheet("font-size: 11px; color: #666;")
        desc.setWordWrap(True)

        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        layout.addWidget(header)

        # 配置：7 个坐标轴
        self.m2_labels = ['x₁', 'y₁', 'z₁', 'w', 'x₂', 'y₂', 'z₂']
        self.m2_units = ['mm', 'mm', 'mm', '', 'mm', 'mm', 'mm']  # w 无单位

        colors = {
            'meas': '#F67280',  # 测量值 - 粉色
            'filt': '#355C7D'  # 滤波值 - 深蓝
        }

        # 创建 7 个子图
        self.m2_plots = {}
        self.m2_curves = {}
        self.m2_buffers = {}

        grid = QGridLayout()
        grid.setSpacing(5)

        for idx, label in enumerate(self.m2_labels):
            plot = pg.PlotWidget()
            plot.setTitle(f"{label} 轴")
            plot.setLabel('left', f'{label} ({self.m2_units[idx]})')
            plot.setLabel('bottom', '帧')
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setBackground('w')
            plot.setMinimumHeight(100)
            plot.setMaximumHeight(140)

            key_meas = f"{label}_meas"
            key_filt = f"{label}_filt"

            self.m2_buffers[key_meas] = np.zeros(self.history_length)
            self.m2_buffers[key_filt] = np.zeros(self.history_length)

            self.m2_curves[key_meas] = plot.plot(
                pen=pg.mkPen(color=colors['meas'], width=1),
                name='Measurement'
            )
            self.m2_curves[key_filt] = plot.plot(
                pen=pg.mkPen(color=colors['filt'], width=2, style=Qt.DashLine),
                name='Filtered'
            )

            plot.addLegend()

            # 特殊处理 w 轴：固定范围 [-1, 1]（如果是四元数）
            if label == 'w':
                plot.setYRange(-1.2, 1.2)
                plot.enableAutoRange(y=False)

            # 按 3 列布局：7 个子图 → 3 行×3 列
            row, col = divmod(idx, 3)
            grid.addWidget(plot, row, col)
            self.m2_plots[label] = plot

        # 使用 ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(QWidget())
        scroll.widget().setLayout(grid)
        layout.addWidget(scroll)

        # 添加标签页
        self.tab_widget.addTab(module2_widget, "🔸 关联位置滤波")

    # ─────────────────────────────────────────────────────────────
    # 数据更新接口
    # ─────────────────────────────────────────────────────────────
    def update_data(self, measurement_m1=None, filtered_m1=None,
                    measurement_m2=None, filtered_m2=None):
        """
        接收新数据并更新可视化（支持单模块或双模块更新）

        Parameters:
        -----------
        measurement_m1 : np.ndarray, shape (2, 3), optional
            模块 1 测量值 [ [xA,yA,zA], [xB,yB,zB] ]
        filtered_m1 : np.ndarray, shape (2, 3), optional
            模块 1 滤波结果
        measurement_m2 : np.ndarray, shape (7,), optional
            模块 2 测量值 [x₁, y₁, z₁, w, x₂, y₂, z₂]
        filtered_m2 : np.ndarray, shape (7,), optional
            模块 2 滤波结果
        """
        if not self.is_updating:
            return

        # 降频检查（如果需要）
        # self.internal_frame_counter += 1
        # if self.internal_frame_counter % self.internal_skip_frames != 0:
        #     return

        # 缓存数据（定时器统一渲染）
        if measurement_m1 is not None and filtered_m1 is not None:
            self.pending_data['module1'] = (measurement_m1, filtered_m1)

        if measurement_m2 is not None and filtered_m2 is not None:
            self.pending_data['module2'] = (measurement_m2, filtered_m2)

    def _render_pending_data(self):
        """定时器触发：渲染缓存的数据"""
        # 渲染模块 1
        if self.pending_data['module1']:
            meas, filt = self.pending_data['module1']
            self._update_module1(meas, filt)
            self.pending_data['module1'] = None

        # 渲染模块 2
        if self.pending_data['module2']:
            meas, filt = self.pending_data['module2']
            self._update_module2(meas, filt)
            self.pending_data['module2'] = None

    def _update_module1(self, measurement, filtered_state):
        """更新模块 1：独立位置滤波（6 子图）"""
        try:
            meas_A, meas_B = measurement[0], measurement[1]
            filt_A, filt_B = filtered_state[0], filtered_state[1]
        except Exception as e:
            print(f"⚠️ 模块 1 数据解析错误：{e}")
            return

        axes = ['X', 'Y', 'Z']
        points_data = {
            'A': {'meas': meas_A, 'filt': filt_A},
            'B': {'meas': meas_B, 'filt': filt_B}
        }

        for i, axis in enumerate(axes):
            for point, data in points_data.items():
                key_meas = f"{point}_{axis}_meas"
                key_filt = f"{point}_{axis}_filt"

                # 滚动缓冲区
                self.m1_buffers[key_meas] = np.roll(self.m1_buffers[key_meas], -1)
                self.m1_buffers[key_filt] = np.roll(self.m1_buffers[key_filt], -1)
                self.m1_buffers[key_meas][-1] = data['meas'][i]
                self.m1_buffers[key_filt][-1] = data['filt'][i]

                # 更新曲线
                x_data = np.arange(self.frame_count - self.history_length + 1, self.frame_count + 1)
                self.m1_curves[key_meas].setData(x_data, self.m1_buffers[key_meas])
                self.m1_curves[key_filt].setData(x_data, self.m1_buffers[key_filt])

    def _update_module2(self, measurement, filtered_state):
        """更新模块 2：关联位置滤波（7 子图）"""
        try:
            # 确保形状正确
            meas = np.asarray(measurement).reshape(7)
            filt = np.asarray(filtered_state).reshape(7)
        except Exception as e:
            print(f"⚠️ 模块 2 数据解析错误：{e}")
            return

        for idx, label in enumerate(self.m2_labels):
            key_meas = f"{label}_meas"
            key_filt = f"{label}_filt"

            # 滚动缓冲区
            self.m2_buffers[key_meas] = np.roll(self.m2_buffers[key_meas], -1)
            self.m2_buffers[key_filt] = np.roll(self.m2_buffers[key_filt], -1)
            self.m2_buffers[key_meas][-1] = meas[idx]
            self.m2_buffers[key_filt][-1] = filt[idx]

            # 更新曲线
            x_data = np.arange(self.frame_count - self.history_length + 1, self.frame_count + 1)
            self.m2_curves[key_meas].setData(x_data, self.m2_buffers[key_meas])
            self.m2_curves[key_filt].setData(x_data, self.m2_buffers[key_filt])

        self.frame_count += 1

    # ─────────────────────────────────────────────────────────────
    # 控制功能
    # ─────────────────────────────────────────────────────────────
    def toggle_update(self):
        """暂停/继续更新"""
        self.is_updating = not self.is_updating
        self.btn_toggle.setText("⏸️ 暂停/继续" if self.is_updating else "▶️ 继续")
        print(f"{'✅' if self.is_updating else '⏸️'} 可视化已{'继续' if self.is_updating else '暂停'}")

    def reset_all_views(self):
        """重置所有数据缓冲区"""
        # 模块 1
        for key in self.m1_buffers:
            self.m1_buffers[key] = np.zeros(self.history_length)
        # 模块 2
        for key in self.m2_buffers:
            self.m2_buffers[key] = np.zeros(self.history_length)

        self.frame_count = 0

        # 重置曲线显示
        x_data = np.arange(-self.history_length + 1, 1)
        for curve in list(self.m1_curves.values()) + list(self.m2_curves.values()):
            curve.setData(x_data, np.zeros(self.history_length))

        print("🔄 所有视图已重置")

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.viz_timer.stop()
        event.accept()