
import time

import numpy as np
import pyvista
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSpinBox
from scipy.spatial import cKDTree
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from view3D.dicom_load_funcs import volume_to_point_cloud, load_ct_voxel_file, resample_to_isotropic
import matplotlib.colors as mcolors

def remove_outliers_by_knn(points, k=10, std_ratio=2.0):
    """
    使用 cKDTree 快速剔除离群点（CPU 高效）
    """
    n_points = len(points)
    if n_points <= k + 1:
        return np.ones(n_points, dtype=bool)

    # 构建 KD 树
    tree = cKDTree(points)
    # 查询每个点的 k+1 个最近邻（包括自己）
    distances, _ = tree.query(points, k=k + 1, workers=-1)  # workers=-1 使用所有 CPU 核
    # 去掉第 0 列（到自身的距离 = 0）
    mean_distances = distances[:, 1:].mean(axis=1)

    # 统计阈值
    mean_dist = np.mean(mean_distances)
    std_dist = np.std(mean_distances)
    threshold = mean_dist + std_ratio * std_dist

    return mean_distances < threshold

class VoxelLoadClipWidget(QWidget):


    def __init__(self, parent_widget):

        self._guide_actor_names = []
        self._latest_sz_actor_names = []
        self._latest_sc_actor_names = []
        # 创建 PyVista Qt 渲染器并嵌入 parent_widget
        self.plotter = QtInteractor(parent_widget)
        self.plotter.ren_win.SetMultiSamples(4)
        layout = QVBoxLayout(parent_widget)
        layout.addWidget(self.plotter)
        parent_widget.setLayout(layout)

        # self.show_spine(voxel_path)


    def show_spine(self, voxel_path, threshold=70):
        ct_vox, vox_space = load_ct_voxel_file(voxel_path)
        origin = ct_vox.shape * vox_space / 2 * (-1)
        ct2ct_centre = np.array([[-1, 0.0000, 0.0000, 0],
                                 [0.0000, -1, 0.0000, 0],
                                 [0.0000, 0.0000, 1.0000, 0],
                                 [0.0000, 0.0000, 0.0000, 1.0000]])


        grid = pyvista.ImageData(
            dimensions=ct_vox.shape, spacing=vox_space, origin=origin
        )

        grid.point_data["values"] = (
                ct_vox.flatten(order="F") > threshold
        )

        mesh = grid.contour_labels(smoothing=True, progress_bar=True)
        mesh = mesh.transform(ct2ct_centre, inplace=True)

        self.plotter.add_mesh(mesh, color='lightgray')
        self.plotter.show()

    # def update_show_spine(self, voxel_path, threshold):
    #     ct_vox, vox_space = load_ct_voxel_file(voxel_path)
    #
    #     grid = pyvista.ImageData(
    #         dimensions=ct_vox.shape, spacing=vox_space, origin=(0, 0, 0)
    #     )
    #
    #     grid.point_data["values"] = (
    #             ct_vox.flatten(order="F") > threshold
    #     )
    #
    #     mesh = grid.contour_labels(smoothing=True, progress_bar=True)
    #
    #     self.plotter.add_mesh(mesh, color='lightgray')
    #     self.plotter.show()
    # def enhanced_show_spine(self, points, intensities, vox_space=None):
    #     """
    #     增强版脊椎点云渲染函数
    #     :param points: (N, 3) 物理坐标（mm）
    #     :param intensities: (N,) 归一化强度 [0, 1]
    #     :param vox_space: [dx, dy, dz]，用于动态设置 point_size（可选）
    #     """
    #     self.plotter.clear()
    #     self.plotter.set_background("black")
    #
    #     if len(points) == 0:
    #         print("No points to display.")
    #         return
    #     print("len(points):", len(points))
    #     # —————— 1. 离群点剔除 ——————
    #     if len(points) > 600_000:
    #
    #         # 随机下采样
    #         idx = np.random.choice(len(points), 600_000, replace=False)
    #         points_sub = points[idx]
    #         intensities_sub = intensities[idx]
    #         # 对子集做离群剔除
    #         valid_mask_sub = remove_outliers_by_knn(points_sub)
    #         # 得到最终 clean 点云（来自子集）
    #         points_clean = points_sub[valid_mask_sub]
    #         intensities_clean = intensities_sub[valid_mask_sub]
    #     else:
    #         valid_mask = remove_outliers_by_knn(points)
    #         points_clean = points[valid_mask]
    #         intensities_clean = intensities[valid_mask]
    #
    #     if len(points_clean) == 0:
    #         print("All points filtered out.")
    #         return
    #
    #     # —————— 2. 构建 PolyData 并估计法向 ——————
    #     cloud = pv.PolyData(points_clean)
    #     cloud["intensity"] = np.clip(intensities_clean, 0.2, 1.0)
    #
    #     # 法向估计（显著提升 lighting 效果）
    #     try:
    #         cloud.compute_normals(
    #             point_normals=True,
    #             cell_normals=False,
    #             feature_angle=30.0,
    #             inplace=True
    #         )
    #     except Exception as e:
    #         print(f"Normal estimation failed (using default normals): {e}")
    #
    #     # —————— 3. 动态 point_size（基于体素间距） ——————
    #     point_size = 8.0 # 默认
    #     if vox_space is not None:
    #         avg_spacing = float(np.mean(vox_space))
    #         point_size = max(1.0, avg_spacing * 1.2)  # 确保点略大于体素
    #
    #     # —————— 4. 渲染 ——————
    #     # 创建从浅米黄到深米黄的 colormap
    #     BEIGE_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    #         "beige_intensity",
    #         [(0.48, 0.42, 0.34), (0.64, 0.58, 0.50), (0.80, 0.74, 0.66)]
    #     )
    #     # beige_color =  (163/255, 148/255, 128/255)
    #     self.plotter.add_mesh(
    #         cloud,
    #         scalars="intensity",
    #         cmap=BEIGE_COLORMAP,
    #         # color=beige_color,
    #         point_size=point_size,
    #         render_points_as_spheres=True,
    #         lighting=True,
    #         ambient=0.3,
    #         diffuse=0.7,
    #         specular=0.4,
    #         specular_power=25,
    #         smooth_shading=True,
    #         opacity=1.0,
    #         show_scalar_bar=False  # 可选：隐藏色标以聚焦模型
    #     )
    #
    #     # —————— 5. 后处理渲染设置 ——————
    #     self.plotter.enable_depth_peeling()  # 改善重叠透明
    #     # self.plotter.enable_anti_aliasing()   # 若 PyVista 版本支持
    #
    #     self.plotter.reset_camera()


    def show_selected_point(self, point):
        """显示选中的点"""
        # 添加点到当前 plotter（假设 self.plotter 是 QtInteractor）
        point_array = np.array([point])
        self.plotter.add_points(
            point_array,
            color='red',  # 颜色（可选）
            point_size=10,  # 点大小
            render_points_as_spheres=True,  # 渲染为球体（更美观）
            name='selected_point'  # 命名以便后续更新/删除
        )

    def show_line_in_ct(self, oct_source, pct, view_type, color):
        """
        view_type: 'sz' (正位) or 'sc' (侧位)
        """
        oct_source = np.asarray(oct_source)
        pct = np.asarray(pct)

        # 1. 清除同类型的旧射线
        if view_type == 'sz':
            actor_names_to_clear = self._latest_sz_actor_names
            self._latest_sz_actor_names = []
        else:
            actor_names_to_clear = self._latest_sc_actor_names
            self._latest_sc_actor_names = []

        for name in actor_names_to_clear:
            self.plotter.remove_actor(name, render=False)

        # 2. 生成新名称
        unique_id = str(int(time.time() * 1000))
        line_name = f'guide_line_{view_type}_{unique_id}'
        source_name = f'guide_source_{view_type}_{unique_id}'
        proj_name = f'guide_proj_{view_type}_{unique_id}'

        # 3. 记录新名称
        new_names = [line_name, source_name, proj_name]
        if view_type == 'sz':
            self._latest_sz_actor_names = new_names
        else:
            self._latest_sc_actor_names = new_names

        # ✅ 保存当前相机状态（关键！）
        camera = self.plotter.camera.copy()

        # # 4. 绘制
        # pct = np.array([-pct[0], pct[1], -pct[2]])
        # oct_source = np.array([pct[0], oct_source[1], pct[2]])
        print('pct, oct_source',pct, oct_source)
        line = pv.Line(oct_source, pct)
        self.plotter.add_mesh(line, color=color, line_width=1.5, name=line_name)

        # 光源点（绿色）

        self.plotter.add_points(
            oct_source, color='green', point_size=6,
            render_points_as_spheres=True, name=source_name
        )
        # 投影点
        self.plotter.add_points(
            pct, color=color, point_size=6,
            render_points_as_spheres=True, name=proj_name
        )

        # ✅ 恢复相机状态（防止自动缩放）
        self.plotter.camera = camera
        self.plotter.add_axes()
        self.plotter.render()


    def show_pin_in_ct(self, real_dire_in_ct, real_pin_in_ct):
        """
        在3D视图中画出手术针的延长射线：
        - real_dire_in_ct: 针尾（起点）
        - real_pin_in_ct: 针尖（用于确定方向）
        """
        real_dire_in_ct = np.asarray(real_dire_in_ct, dtype=float)
        real_pin_in_ct = np.asarray(real_pin_in_ct, dtype=float)

        # 计算方向向量（针尾 → 针尖）
        direction = real_pin_in_ct - real_dire_in_ct
        norm = np.linalg.norm(direction)
        if norm == 0:
            return  # 两点重合，不画线

        direction = direction / norm

        # 射线终点：从针尾沿方向延伸 200mm
        ray_end = real_dire_in_ct + direction * 400.0

        camera = self.plotter.camera.copy()

        # 绘制射线（针尾 → 延长终点）
        ray_line = pv.Line(real_dire_in_ct, ray_end)
        self.plotter.add_mesh(
            ray_line,
            color='cyan',
            line_width=2,
            name='surgical_pin_ray'
        )

        # （可选）高亮针尾和针尖
        points = pv.PolyData(np.vstack([real_dire_in_ct, real_pin_in_ct]))
        self.plotter.add_mesh(
            points,
            color='yellow',
            point_size=6,
            render_points_as_spheres=True,
            name='pin_points'
        )

        self.plotter.camera = camera
        self.plotter.render()


    def clear_all_guide_lines(self):
        all_names = self._latest_sz_actor_names + self._latest_sc_actor_names
        for name in all_names:
            self.plotter.remove_actor(name, render=False)
        self._latest_sz_actor_names = []
        self._latest_sc_actor_names = []
        self.plotter.render()