import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


from projector.pose import convert
from projector.post_processing import normalize_to_255, apply_circular_mask
from projector.drr import DRR
from projector.read_data import read
from projector.visualization import plot_drr
from PIL import Image

# Create filename based on rotation and translation parameters
def create_filename(rot, trans):
    # Format: drr_rotX_Y_Z_transX_Y_Z_timestamp.png
    rot_str = "_".join([f"{r:.1f}" for r in rot.squeeze().tolist()])
    trans_str = "_".join([f"{t:.1f}" for t in trans.squeeze().tolist()])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"drr_rot{rot_str}_trans{trans_str}_{timestamp}.png"

def euler_to_rotvec(euler_deg):
    """ZXY 欧拉角 → 旋转向量（角度制）"""
    R = euler_angles_to_rotation_matrix_zxy(euler_deg)
    # 通过矩阵对数映射获取旋转角和轴
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    axis = axis / (2 * np.sin(angle))
    return np.rad2deg(angle) * axis  # 旋转向量，单位：度

def rotvec_to_euler(rotvec_deg):
    """旋转向量 → ZXY 欧拉角"""
    angle = np.deg2rad(np.linalg.norm(rotvec_deg))
    if angle < 1e-6:
        return np.zeros(3)
    axis = rotvec_deg / np.linalg.norm(rotvec_deg)
    # 构建旋转矩阵（Rodrigues 公式）
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K
    return rotation_matrix_to_euler_angles_zxy_numpy(R)



def euler_angles_to_rotation_matrix_zxy(euler_angles_deg):
    """ZXY 欧拉角转旋转矩阵"""
    theta_z, theta_x, theta_y = np.deg2rad(euler_angles_deg)
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)

    R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

    return R_y @ R_x @ R_z


def rotation_matrix_to_euler_angles_zxy_numpy(R):
    """旋转矩阵转 ZXY 欧拉角"""
    sin_x = -np.clip(R[1, 2], -1.0, 1.0)
    theta_x = np.arcsin(sin_x)
    theta_z = np.arctan2(R[1, 0], R[1, 1])
    theta_y = np.arctan2(R[0, 2], R[2, 2])
    return np.rad2deg([theta_z, theta_x, theta_y])


def get_transformer_noise_inverse(rot_noise_new, trans_noise_new, R_ct2o):
    """
    逆变换：OSZ 系等效噪声 → CT 系原始噪声

    Args:
        rot_noise_new: [3] OSZ 系旋转噪声欧拉角 (度)
        trans_noise_new: [3] OSZ 系平移噪声 (毫米)
        R_ct2o: [4, 4] CT→OSZ 齐次变换矩阵

    Returns:
        rotations: [3] CT 系旋转噪声欧拉角 (度)
        translations: [3] CT 系平移噪声 (毫米)
    """
    rotations = np.array(rot_noise_new, dtype=np.float64)
    translations = np.array(trans_noise_new, dtype=np.float64)
    R_ct2o = np.array(R_ct2o, dtype=np.float64)

    R_ct2o_rot = R_ct2o[:3, :3]

    # ================= 1. 旋转逆变换（相似变换的逆）=================
    # OSZ 系欧拉角 → 旋转矩阵
    R_noise_osz = euler_angles_to_rotation_matrix_zxy(rotations)

    # 相似变换的逆：R_ct = R_ct2o.T @ R_osz @ R_ct2o
    R_noise_ct = R_ct2o_rot.T @ R_noise_osz @ R_ct2o_rot

    # 旋转矩阵 → CT 系欧拉角
    rotations_ct = rotation_matrix_to_euler_angles_zxy_numpy(R_noise_ct)

    # ================= 2. 平移逆变换（向量变换的逆）=================
    # translations_ct = R_ct2o.T @ trans_noise_osz
    translations_ct = R_ct2o_rot.T @ translations

    return rotations_ct, translations_ct

def get_transformer_noise(rotations, translations, R_ct2o):
    """
    计算相对于 O 的等效噪声（修正版）
    使用旋转矩阵相似变换，保证几何一致性
    """

    R_ct2o_rot = R_ct2o[:3, :3]

    # ================= 1. 旋转变换（相似变换）⭐ =================
    # 1.1 CT 噪声欧拉角 -> 旋转矩阵
    R_noise_ct = euler_angles_to_rotation_matrix_zxy(rotations)

    # 1.2 相似变换：R_osz = R_ct2osz @ R_ct @ R_ct2osz.T
    R_noise_osz = R_ct2o_rot @ R_noise_ct @ R_ct2o_rot.T

    # 1.3 旋转矩阵 -> OSZ 噪声欧拉角
    rot_noise_new = rotation_matrix_to_euler_angles_zxy_numpy(R_noise_osz)

    # ================= 2. 平移变换（向量变换）=================
    trans_noise_new = R_ct2o_rot @ translations

    return rot_noise_new, trans_noise_new


def get_transformer_noise_rotvec(rotations, translations, R_ct2o):
    """
    基于旋转向量的噪声变换 (推荐方案)
    """
    R_ct2o_rot = R_ct2o[:3, :3]

    # ================= 1. 旋转变换 =================
    # 1.1 欧拉角 -> 旋转向量 (3D 向量，模长=角度)
    rot_vec_ct = euler_to_rotvec(rotations)

    # 1.2 核心变换：直接旋转该向量 (线性变换)
    # 物理意义：旋转轴随坐标系旋转，旋转角度保持不变
    rot_vec_osz = R_ct2o_rot @ rot_vec_ct

    # 1.3 旋转向量 -> 欧拉角 (仅在需要输出欧拉角时转换)
    rot_noise_new = rotvec_to_euler(rot_vec_osz)

    # ================= 2. 平移变换 =================
    # 平移依然是向量变换，保持不变
    trans_noise_new = R_ct2o_rot @ translations

    return rot_noise_new, trans_noise_new

def verify_ct1_ct2_independent(rotations_ct, rotations_o, R_ct2o, test_points=None):
    """
    使用独立坐标变换链验证 ct1 和 ct2 一致性
    不依赖相似变换公式本身
    """

    if test_points is None:
        # 生成一组测试点（立方体顶点 + 随机点）
        test_points = np.array([
            [0, 0, 0],  # 原点
            [1, 0, 0],  # X 轴
            [0, 1, 0],  # Y 轴
            [0, 0, 1],  # Z 轴
            [1, 1, 1],  # 对角点
            [10, -5, 20],  # 随机点
            [-3, 7, -15],  # 随机点
        ], dtype=np.float64)

    R_ct2o_rot = R_ct2o[:3, :3]

    # 构建旋转矩阵
    R_ct = euler_angles_to_rotation_matrix_zxy(rotations_ct)
    R_o = euler_angles_to_rotation_matrix_zxy(rotations_o)

    print("=" * 70)
    print("独立坐标变换链验证（不依赖相似变换公式）")
    print("=" * 70)

    all_match = True
    for i, p in enumerate(test_points):
        # # ========== 路径 1: CT 系直接变换 ==========
        # # CT 系点 → 施加 CT 噪声旋转 → 转换到 OSZ 系
        # p_ct_rotated = R_ct @ p  # CT 系内旋转
        # p_ct_to_osz = R_ct2osz_rot @ p_ct_rotated  # CT→OSZ 坐标变换
        #
        # # ========== 路径 2: OSZ 系变换 ==========
        # # CT 系点 → 转换到 OSZ 系 → 施加 OSZ 噪声旋转
        # p_to_osz = R_ct2osz_rot @ p  # CT→OSZ 坐标变换
        # p_osz_rotated = R_osz @ p_to_osz  # OSZ 系内旋转
        p_ct_rotated = R_ct @ p
        p_to_o = R_ct2o_rot @ p  # CT→OSZ 坐标变换
        p_o_rotated = R_o @ p_to_o  # OSZ 系内旋转
        p_o_to_ct = R_ct2o_rot.T @ p_o_rotated  # OSZ→CT 坐标变换

        # ========== 比较两条路径的结果 ==========
        error = np.max(np.abs(p_o_to_ct - p_ct_rotated))
        match = np.allclose(p_o_to_ct, p_ct_rotated, atol=1e-10)
        all_match = all_match and match

        # === 修复：将 numpy 数组转换为列表或字符串再格式化 ===
        p_str = np.array2string(p, precision=1, separator=', ')
        p1_str = np.array2string(p_o_to_ct.round(4), precision=4, separator=', ')
        p2_str = np.array2string(p_ct_rotated.round(4), precision=4, separator=', ')

        print(f"点 {i + 1}: {p_str:18} → 路径 1:{p1_str:22} | "
              f"路径 2:{p2_str:22} | 误差:{error:.2e} {'✅' if match else '❌'}")

    print("\n" + "=" * 70)
    if all_match:
        print("✅ 两条独立路径结果一致！ct1 和 ct2 空间姿态完全等价。")
        print("   这证明了 R_o 的计算是正确的（不依赖相似变换自证）。")
    else:
        print("❌ 两条路径结果不一致，需要检查 R_o 的计算。")
    print("=" * 70)

    return all_match



if __name__ == '__main__':
    # Read in the volume and get its origin and spacing in world coordinates
    # subject = load_example_ct(orientation="AP")
    # volume_dir = "data/CT25/lum_25.nii.gz"
    volume_dir_2 = r"../data/voxel_data/spine107_img.nii.gz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 生成与光源坐标系一致的噪声
    # rotations = np.array([5, 5, 5], dtype=np.float64)
    # translations = np.array([1, 2, 3], dtype=np.float64)

    rot_label = np.array([-1.075522,0.215987,-4.190782], dtype=np.float64)
    trans_label = np.array([21.910440,-4.974796,-24.264019], dtype=np.float64)


    # subject_pa = read(volume_dir_2, bone_attenuation_multiplier=1.0, orientation='PA', sid=500)
    # drr_pa = DRR(
    #     subject_pa,  # An object storing the CT volume, origin, and voxel spacing
    #     sdd=800,  # Source-to-detector distance (i.e., focal length)
    #     height=512,  # Image height (if width is not provided, the generated DRR is square)
    #     delx=0.469,  # Pixel spacing (in mm)
    #     renderer="trilinear"
    # ).to(device)
    # # PA视角噪声变换矩阵(旋转变换)
    # R_ct2osz = np.array([[-1, 0, 0, 0],
    #            [0, 0, 1, 0],
    #            [0, 1, 0, 0],
    #            [0, 0, 0, 1]],dtype=np.float64)
    #
    # rot_noise_new_PA, trans_noise_new_PA = get_transformer_noise(rotations, translations, R_ct2osz)
    # print("PA-label:",rot_noise_new_PA,trans_noise_new_PA)
    # rot_noise_new_PA = rot_label
    # trans_noise_new_PA = trans_label
    # rotations_ct, translations_ct = get_transformer_noise_inverse(rot_noise_new_PA, trans_noise_new_PA, R_ct2osz)
    # print("PA-noise:", rotations_ct, translations_ct)
    # verify_ct1_ct2_independent(rotations, rot_noise_new_PA, R_ct2osz)


    # subject_rlat = read(volume_dir_2, bone_attenuation_multiplier=1.0, orientation='RLAT', sid=500)
    # drr_rlat = DRR(
    #     subject_rlat,  # An object storing the CT volume, origin, and voxel spacing
    #     sdd=800,  # Source-to-detector distance (i.e., focal length)
    #     height=512,  # Image height (if width is not provided, the generated DRR is square)
    #     delx=0.469,  # Pixel spacing (in mm)
    #     renderer="trilinear"
    # ).to(device)

    # # RLAT视角噪声变换矩阵(旋转变换)
    R_ct2osc = np.array([[0, -1, 0, 0],
                         [0, 0, 1, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 0, 1]], dtype=np.float64)
    # rot_noise_new_RLAT, trans_noise_new_RLAT = get_transformer_noise(rotations, translations, R_ct2osc)
    # print("RLAT-label:", rot_noise_new_RLAT, trans_noise_new_RLAT)
    rot_noise_new_RLAT = rot_label
    trans_noise_new_RLAT = trans_label
    rotations_ct, translations_ct = get_transformer_noise_inverse(rot_noise_new_RLAT, trans_noise_new_RLAT, R_ct2osc)
    print("RLAT-noise:", rotations_ct, translations_ct)
    # verify_ct1_ct2_independent(rotations, rot_noise_new_RLAT, R_ct2osc)

    # rotations_t = torch.tensor(rotations, dtype=torch.float32, device=device)
    # translations_t = torch.tensor(translations, dtype=torch.float32, device=device)
    # rotations_t = torch.tensor(rotations, dtype=torch.float32, device=device)
    # translations_t = torch.tensor(translations, dtype=torch.float32, device=device)
    #
    # === 方案 1: 使用 unsqueeze 添加 Batch 维度 ===
    # if rotations_t.dim() == 1:
    #     rotations_t = rotations_t.unsqueeze(0)  # [3] -> [1, 3]
    # if translations_t.dim() == 1:
    #     translations_t = translations_t.unsqueeze(0)  # [3] -> [1, 3]
    #
    # img = drr_pa(rotations_t, translations_t, parameterization="euler_angles", convention="ZXY", degrees=True)
    # # img = drr_rlat(rotations_t, translations_t, parameterization="euler_angles", convention="ZXY", degrees=True)
    # img = normalize_to_255(img)
    #
    # mask_img = apply_circular_mask(img)
    # plot_drr(mask_img, ticks=False)
    #
    # plt.show()