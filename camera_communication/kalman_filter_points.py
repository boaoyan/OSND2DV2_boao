import numpy as np

from camera_communication.utils.ESKF_filter_points import ESKF_1step_2Points



class KalmanFilterPoints:


    def __init__(self, d=114, sigma_Q=None, sigma_R=2.0, init_P_scale=1000.0):
        """
        初始化实时 ESKF 滤波器（用于两个刚性连接点）

        Parameters:
        ----------
        d : float
            两点间固定距离（单位：mm）
        sigma_Q : array-like, [sigma_trans, sigma_rot]
            过程噪声标准差（每步），例如 [0.1, 0.00175]
        sigma_R : float
            测量噪声标准差（各坐标轴，单位：mm）
        init_P_scale : float
            初始协方差缩放因子（默认 1000）
        """
        self.d = d
        self.sigma_Q = np.array([5 / 50, 5 * np.pi / (180 * 50)]) if sigma_Q is None else np.asarray(sigma_Q)
        self.sigma_R = sigma_R
        self.P = init_P_scale * np.eye(5)  # (5,5)
        self.state_6d = None  # (6,) current filtered [pA; pB]
        self.initialized = False

    def update(self, measurement):
        """
        接收一帧新测量，执行 ESKF 更新

        Parameters:
        ----------
        measurement : array-like, shape (6,)
            当前帧测量值 [xA, yA, zA, xB, yB, zB]

        Returns:
        -------
        filtered_state : np.ndarray, shape (6,)
            滤波后的 [pA; pB]
        """
        measurement = np.asarray(measurement, dtype=np.float64)

        # 记录输入形状，用于输出还原
        input_shape = measurement.shape
        # print(f"[DEBUG] Input shape: {input_shape}")
        # 转换为 (6,) 向量供内部使用
        if input_shape == (2, 3):
            meas_6d = measurement.reshape(6)
        elif input_shape == (6,):
            meas_6d = measurement
        else:
            raise ValueError(f"Measurement must be shape (6,) or (2, 3), got {input_shape}")

        if not self.initialized:
            # 第一帧：直接作为初始状态
            self.state_6d = meas_6d.copy()
            self.initialized = True
            return self.state_6d.copy().reshape(input_shape)

        # 执行 ESKF 一步
        self.state_6d, self.P = ESKF_1step_2Points(
            self.state_6d, self.P, meas_6d,
            self.d, self.sigma_Q, self.sigma_R
        )

        result = self.state_6d.copy().reshape(input_shape)
        # print(f"[DEBUG] Output shape: {result.shape}")  # ← 加这行
        return result


    # def show_viewer(self):
    #     """显示可视化界面"""
    #     if self.viewer is None:
    #         self.viewer = TimeSeriesViewer(self.data_manager)
    #     self.viewer.show()
    #     self.viewer.raise_()
    #     self.viewer.activateWindow()