"""
已知目标点位和电机中工具点到相机中的对应关系
求最佳电机旋转角度AB，使得目标点到电机中工具点所在的投影线的距离最小
传入的所有点都是相机坐标系下的点，传出的是电机应该旋转到的位置
"""
import numpy as np
from scipy.optimize import minimize
from robot_arm.utils.cali_utils import transform_points
from robot_arm.utils.tool_to_arm import get_tool2arm_rt


def get_distance_to_line(P0, P1, P2):
    # 方向向量
    line_vec = P2 - P1
    # 从直线上一点到目标点的向量
    point_vec = P0 - P1

    # 叉积法求点到直线距离
    cross_prod = np.cross(line_vec, point_vec)
    distance = np.linalg.norm(cross_prod) / (np.linalg.norm(line_vec) + 1e-12)  # 防除零
    return distance


class IterateMoveControl:
    def __init__(self, config: dict):
        self.a_threshold = config['a_threshold']
        self.b_threshold = config['b_threshold']
        self.param_toolP = np.array(config['param_toolP'])
        self.param_cz = np.array(config['param_cz'])
        self.init_a = config['init_a']
        self.init_b = config['init_b']
        self.rt_arm2cam = None

    def objective(self, x, aim_in_cam):
        a_pos, b_pos = x
        a_pos = ((float(a_pos) - self.init_a) / 2048) * np.pi
        b_pos = ((float(b_pos) - self.init_b) / 2048) * np.pi

        # Step 1: 计算 toolP 在 arm 坐标系下的位置
        tool2arm_r, tool2arm_t = get_tool2arm_rt(a_pos, b_pos, self.param_cz)
        param_toolP = self.param_toolP.T
        ball_in_arm = tool2arm_r @ param_toolP + tool2arm_t

        # Step 2: 变换到相机坐标系
        arm2cam_r = self.rt_arm2cam[:3, :3]
        arm2cam_t = self.rt_arm2cam[:3, 3].reshape(3, 1)
        ball_in_cam = arm2cam_r @ ball_in_arm + arm2cam_t # (2,3)
        ball_in_cam = ball_in_cam.T
        P1, P2 = ball_in_cam[0], ball_in_cam[1]

        # Step 3: 计算 aim_in_cam 到直线 P1-P2 的距离
        distance = get_distance_to_line(aim_in_cam, P1, P2)
        return distance

    def control_to_aim(self, aim_in_cam, x0):
        """
        aim_in_cam: (3,) 目标点，3D坐标，在相机坐标系下
        返回：最优 a_pos, b_pos（电机角度）
        """
        if self.rt_arm2cam is None:
            raise ValueError("rt_arm2cam is None, 无法进行控制")

        # 初始猜测值
        x0 = x0

        # 边界约束
        bounds = [
            (self.a_threshold[0], self.a_threshold[1]),
            (self.b_threshold[0], self.b_threshold[1])
        ]

        # 执行优化
        result = minimize(
            fun=self.objective,
            x0=x0,
            args=(aim_in_cam,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False, 'maxiter': 150}
        )

        if result.success:
            optimal_a, optimal_b = result.x
            print("优化成功: a={}, b={}".format(optimal_a, optimal_b))
            return float(optimal_a), float(optimal_b)
        else:
            print("⚠️ 优化失败:", result.message)
            return float(x0[0]), float(x0[1])  # 返回初始值或报错处理
