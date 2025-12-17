import os
import numpy as np
import pydicom
import vtk
from pydicom import dcmread
from vtkmodules.util.numpy_support import numpy_to_vtk


def _slices2voxels(slices):
    """
    因为slice的序号与z轴是反向的，所以在实际导入的时候需要逆序导入到ct_vox当中
    :param slices:
    :return:
    """
    # create 3D array
    ct_vox = np.zeros([slices[0].pixel_array.shape[0], slices[0].pixel_array.shape[1], len(slices)], dtype=np.int16)
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img = s.pixel_array
        ct_vox[:, :, -i - 1] = img.T.astype(np.int16)
    print("HU range:", ct_vox.min(), ct_vox.max())
    return ct_vox

def _slices2voxels_updata(slices):
    """
    构建 3D 体数据，保持 slices 的原始顺序（应已按物理位置正确排序）
    """
    ct_vox = np.stack([s.pixel_array for s in slices], axis=-1)  # (H, W, D)
    print("HU range:", ct_vox.min(), ct_vox.max())
    return ct_vox.astype(np.int16)

def load_ct_voxel_file(folder):
    """
    Inspired by https://github.com/pydicom/pydicom/blob/master/examples/image_processing/reslice.py
    :param str folder: see proj_ct_to_xray.
    :return tuple(np.ndarray, list(float), list(float), list(float)): ct_img3d, voxel_spacing, position, orientation
    """
    # load the DICOM files
    filenames = [fn for fn in os.listdir(folder) if fn[-4:] == '.dcm']
    print('Loading {} files from {}'.format(len(filenames), folder))

    # skip files with no InstanceNumber
    slices = []
    for f_name in filenames:
        f = pydicom.read_file(os.path.join(folder, f_name), force=True)
        if hasattr(f, 'InstanceNumber'):
            slices.append(f)
            # print(f.ImagePositionPatient)
        else:
            print('File {} has no InstanceNumber'.format(f_name))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda slice: slice.InstanceNumber)
    vox_space = np.array([slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[0].SliceThickness])
    # slices.sort(key=lambda s: s.ImagePositionPatient[2], reverse=True)  # 头→足：Z 递减
    #
    # vox_space = np.array([
    #     float(slices[0].PixelSpacing[0]),
    #     float(slices[0].PixelSpacing[1]),
    #     float(slices[0].SliceThickness)
    # ])
    ct_vox = _slices2voxels_updata(slices)

    # 获取位置和方向（可选，用于严格空间对齐）
    position = list(slices[0].ImagePositionPatient) if hasattr(slices[0], 'ImagePositionPatient') else [0, 0, 0]
    orientation = list(slices[0].ImageOrientationPatient) if hasattr(slices[0], 'ImageOrientationPatient') else [1, 0,
                                                                                                                 0, 0,
                                                                                                   1, 0]
    return ct_vox, vox_space, position, orientation


def resample_to_isotropic(ct_vox, vox_space):
    # 边界置零（修正版）
    ct_vox = ct_vox.copy()
    ct_vox[[0, -1], :, :] = 0
    ct_vox[:, [0, -1], :] = 0
    ct_vox[:, :, [0, -1]] = 0

    new_space = vox_space.copy()
    if not is_close(vox_space[2], vox_space[0]):
        ct_real_size = (np.array(ct_vox.shape) - 1) * vox_space
        rows, cols, page = ct_vox.shape
        z = np.linspace(0, ct_real_size[2], page)
        new_z_spacing = (vox_space[0] + vox_space[1]) / 2
        new_space = np.array([vox_space[0], vox_space[1], new_z_spacing])
        new_page = int(ct_real_size[2] / new_z_spacing)
        new_z = np.linspace(0, ct_real_size[2] - new_z_spacing, new_page)

        iz = np.searchsorted(z, new_z, side='right') - 1
        iz[iz >= page - 1] = page - 2
        dz2_z = z[iz + 1] - new_z
        dz_z1 = new_z - z[iz]
        dz_in_z = z[iz + 1] - z[iz]
        # 防除零
        dz_in_z = np.where(dz_in_z == 0, 1, dz_in_z)
        ct_vox = (ct_vox[:, :, iz + 1] * dz_z1 + ct_vox[:, :, iz] * dz2_z) / dz_in_z

    # 返回 CPU NumPy 数组（即使内部用了 CuPy）
    if hasattr(ct_vox, 'get'):
        ct_vox = ct_vox.get()
    return ct_vox, new_space





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




def load_dicom_volume(folder: str):
    """加载 DICOM 文件夹为 3D 体素数组 (H, W, D) 和体素间距 (mm)"""
    files = [f for f in os.listdir(folder) if f.endswith('.dcm')]
    slices = []
    for f in files:
        ds = dcmread(os.path.join(folder, f))
        if hasattr(ds, 'InstanceNumber'):
            slices.append(ds)
    slices.sort(key=lambda s: s.InstanceNumber)

    # 提取图像和间距
    vox_space = np.array([
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
        float(slices[0].SliceThickness)
    ])
    volume = np.stack([s.pixel_array for s in slices], axis=-1)  # (512, 512, 107)
    return volume.astype(np.float32), vox_space


def volume_to_point_cloud(volume, vox_space, threshold=70, max_points=1_000_000):
    """
    将体素数据转为物理坐标的点云（单位：mm）
    :param volume: np.ndarray, shape (H, W, D)
    :param vox_space: [dx, dy, dz] in mm, e.g., [0.35, 0.35, 1.0]
    :return: points (N,3) in mm, intensities (N,)
    """
    H, W, D = volume.shape
    dx, dy, dz = vox_space  # 0.35, 0.35, 1.0

    # 获取非零（或高于阈值）的体素位置（返回索引）
    y_idx, x_idx, z_idx = np.where(volume > threshold)  # 注意：np.where 返回 (y, x, z)

    # 转为物理坐标（毫米）
    x_mm = x_idx * dx   # X 方向
    y_mm = y_idx * dy   # Y 方向
    z_mm = (D - 1 - z_idx) * dz   # Z 方向

    points = np.stack([x_mm, y_mm, z_mm], axis=1)  # shape (N, 3)
    intensities = volume[y_idx, x_idx, z_idx].astype(np.float32)

    # 归一化灰度
    intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)

    # 下采样（可选）
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        intensities = intensities[idx]

    return points, intensities