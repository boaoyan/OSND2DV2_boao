import pyvista
from view3D.dicom_load_funcs import volume_to_point_cloud, load_ct_voxel_file
voxel_path = "data/spine107_img_cropped"
ct_vox, vox_space, position, orientation = load_ct_voxel_file(voxel_path)

grid = pyvista.ImageData(
        dimensions=ct_vox.shape, spacing=vox_space, origin=(0, 0, 0)
    )


grid.point_data["values"] = (
    ct_vox.flatten(order="F") > 70
)

mesh = grid.contour_labels(smoothing=True, progress_bar=True)
plotter = pyvista.Plotter()
plotter.add_mesh(mesh, color='lightgray')
plotter.show()