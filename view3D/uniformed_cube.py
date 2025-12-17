import numpy as np
import cupy as cp
from main_logic.voxel_process.abstract_base_cube import AbstractBaseCube

def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    判断两个浮点数a和b是否近似相等。

    参数:
    a, b (float): 要比较的两个浮点数。
    rel_tol (float): 相对容差值，默认为1e-09。
    abs_tol (float): 绝对容差值，默认为0.0。

    返回:
    bool: 如果a和b近似相等，则返回True；否则返回False。

    注意:
    相对容差值rel_tol用于处理大小相近的数的比较，而绝对容差值abs_tol用于处理非常小的数的比较。
    只要满足|a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)，则认为a和b近似相等。
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class UniformedCube(AbstractBaseCube):
    def __init__(self):
        super().__init__()
        self.clip_threshold = None

    def update_cube(self, ct_vox, vox_space):
        # 初始化的时候需要把体素的外边一圈置为0
        ct_vox[0, -1] = 0
        ct_vox[:, [0, -1]] = 0
        ct_vox[:, :, [0, -1]] = 0
        new_space = vox_space
        if not is_close(vox_space[2], vox_space[0]):
            # 默认初始重采样体素的z方向的间隔与xy方向的间隔相等
            # 根据毫米单位的体素在z方向的大小和xy方向的间隔来确定初始重采样体素z方向的采样点
            ct_real_size = (np.array(ct_vox.shape) - 1) * vox_space
            rows, cols, page = ct_vox.shape
            z = np.linspace(0, ct_real_size[2], page)
            new_space = np.array([vox_space[0], vox_space[1], (vox_space[0] + vox_space[1]) / 2])
            new_page = int(ct_real_size[2] / ((vox_space[0] + vox_space[1]) / 2))
            new_z = np.linspace(0, ct_real_size[2] - new_space[2], new_page)
            # note 初始化体素仅需沿z轴一维线性插值
            iz = np.searchsorted(z, new_z, side='right') - 1
            # 索引大于最大索引的部分取最大索引-1
            iz[iz >= page - 1] = page - 2
            # 计算每个点在z维度上到邻近点的距离
            dz2_z = z[iz + 1] - new_z
            dz_z1 = new_z - z[iz]
            dz_in_z = 1 / (z[iz + 1] - z[iz])
            ct_vox = (ct_vox[:, :, iz + 1] * dz_z1 + ct_vox[:, :, iz] * dz2_z) / dz_in_z
        # 在初始的时候就转换到GPU上
        if isinstance(ct_vox, np.ndarray):
            ct_vox = cp.array(ct_vox)
        self.cube = ct_vox
        self.vox_space = new_space[0]
        self.clip_threshold = (int(np.min(ct_vox)), int(np.max(ct_vox)))
        return new_space[0]

    @property
    def clip_range(self):
        return self.clip_threshold

    @clip_range.setter
    def clip_range(self, value):
        self.clip_threshold = value

    def get_self_cube(self):
        """
        使用cube只能访问到cube的备份
        使用此函数可以直接访问cube
        :return:
        """
        return self._cube
