import numpy as np

from registration.math_process.data_fusion import euler_angles_to_rotation_matrix_zxy, \
    rotation_matrix_to_euler_angles_zxy_numpy

# 你的数据
pa_rota = np.array([[-0.615322, -3.018059, -2.578065],
                    [ 0.418664, -4.187374,  2.614614],
                    [-0.317884, -1.497839, -3.685053]])

sz_rota = np.array([[172.274656, -0.971212,  3.913945],
                    [-179.801570, 6.469544,  9.548560],
                    [177.032167, -2.852009,  7.141860]])

# 重建旋转矩阵
R_pa = euler_angles_to_rotation_matrix_zxy(pa_rota)
R_sz = euler_angles_to_rotation_matrix_zxy(sz_rota)

# 计算矩阵差异
matrix_error = np.max(np.abs(R_pa - R_sz))
print(f"🔍 旋转矩阵最大元素误差: {matrix_error:.2e}")

# 计算残差旋转的欧拉角（标准误差指标）
R_err = np.einsum('nij,njk->nik', R_pa.transpose(0,2,1), R_sz)
err_euler = rotation_matrix_to_euler_angles_zxy_numpy(R_err)
print(f"📐 残差欧拉角误差 (度):\n{err_euler}")