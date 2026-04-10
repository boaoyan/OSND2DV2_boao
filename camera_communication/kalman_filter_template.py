import numpy as np

from camera_communication.utils.ESKF_filter_template import ESKF_1step_RW


class KalmanFilterTemplate:

    def __init__(self, sigma_Q=None, sigma_R=None):
        """
        初始化 ESKF 滤波器（但不初始化 state/P，等待首帧）
        """
        self.sigma_Q = np.array([1.0, 0.002]) if sigma_Q is None else np.asarray(sigma_Q)
        self.sigma_R = np.array([4.0, 9 * np.pi / 180]) if sigma_R is None else np.asarray(sigma_R)

        self.state = None  # [px, py, pz, qw, qx, qy, qz]
        self.P = None  # (6, 6) 协方差
        self.initialized = False

    def update(self, measure):
        """
        输入一帧观测，返回当前滤波后的 state。
        首次调用时自动初始化。

        参数:
            measure (array-like): (7,) 观测值 [x, y, z, w, x, y, z]

        返回:
            state (np.ndarray): (7,) 滤波后的位姿
        """
        # print("measure: ", measure)
        measure = np.asarray(measure, dtype=np.float64)
        if measure.shape != (7,):
            raise ValueError("measure must be a 7-element vector [x,y,z,w,x,y,z]")

        if not self.initialized:
            # 第一帧：直接作为初始状态
            self.state = measure.copy()
            self.P = 1000 * np.eye(6)
            self.initialized = True
            return self.state.copy()

        # 后续帧：执行 ESKF 更新
        self.state, self.P = ESKF_1step_RW(
            self.state, self.P, measure, self.sigma_Q, self.sigma_R
        )

        # 可选：定期归一化四元数防止漂移
        self.state[3:7] /= np.linalg.norm(self.state[3:7])
        # print("state_filter:", self.state)

        return self.state.copy()


# def kalman_filter(dataSqu, N=100):
#     state = np.concatenate([dataSqu[:3, 0], dataSqu[3:7, 0]])  # [px, py, pz, qw, qx, qy, qz]
#     P = 1000 * np.eye(6)  # 注意：ESKF 通常误差状态为 6D（3位移+3小旋转）
#
#     stateFilted = np.zeros((7, N))
#
#     # 噪声参数（与 MATLAB 一致）
#     sigma_Q = np.array([1.0, 0.002])  # 过程噪声：位置随机游走 std，姿态随机游走 std（rad）
#     sigma_R = np.array([4.0, 9 * np.pi / 180])  # 观测噪声：位置 std (mm?)，姿态 std (rad)
#
#     for k in range(N):
#         measures = dataSqu[:, k]
#
#         # 调用 ESKF 单步更新（需你自己实现此函数）
#         state, P = ESKF_1step_RW(state, P, measures, sigma_Q, sigma_R)
#         stateFilted[:, k] = state
#
#     return state, P




