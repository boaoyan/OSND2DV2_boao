import os

import numpy as np
import torch
import torch.nn.functional as F

from registration.grid_efficientnet.grid_efficientb0_model import GridModel
from registration.label_transform import LabelTransform
from registration.projector.drr import DRR
from registration.projector.read_data import read
from registration.projector.pose import convert
from ui_interaction.ui_response.utils.euler_rotation_transform import rotation_to_euler_angles
from ui_interaction.ui_response.utils.reg_rt_transform import reg_rt_transform


def norm_img(img):
    # img: [B, 1, H, W]
    B, C, H, W = img.shape
    img_flat = img.view(B, -1)  # [B, H*W]

    img_min = img_flat.min(dim=1, keepdim=True).values  # [B, 1]
    img_max = img_flat.max(dim=1, keepdim=True).values  # [B, 1]

    # 避免除零（当 max == min）
    denom = img_max - img_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    img_norm = (img_flat - img_min) / denom
    img_norm = img_norm.view(B, C, H, W)  # [B, 1, 512, 512]
    return img_norm

def init_config():
    return {
        "batch_size": 1,
        "lr": 5e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "height": 224,
        "weight_loss": 1e-2,
        "checkpoint_dir_prefix": "../output/models/loss_in_best/cube_loss",  # 保存路径前缀
        "log_dir_prefix": "../output/logs/loss_in_best/cube_loss",           # 日志路径前缀
        "log_interval": 100,
        "patience": 25,
        "min_delta": 1e-6,
        "max_saved_model_num": 5,
        "val_steps": 25,
        "max_steps": 50,
        "model_config": {
            'edffn' : 1,
            'eca' : 0,
            'conv_mlp': 0
        },
        "noise_params":{
            'standard_pose': 'PA',
            'PA':{
                'trans_noise_range': [25, 25, 25],
                'rota_noise_range': [5, 5, 10],
                'model_path': "../data/registration_model/final_model_PA.pth",
            },
            'RLAT':{
                'trans_noise_range': [25, 25, 25],
                'rota_noise_range': [10, 5, 5],
                'model_path': "../data/registration_model/final_model_RLAT.pth",
            }
        }
    }

def get_mse_loss(config, label_transformer, labels, outputs, pts):
    pre_rota, pre_trans = label_transformer.label2real(outputs)
    tru_rota, tru_trans = label_transformer.label2real(labels)
    tru_pose = convert(tru_rota, tru_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
    pre_pose = convert(pre_rota, pre_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
    tru_pts = torch.matmul(tru_pose.rotation.mT, pts.T)
    pre_pts = torch.matmul(pre_pose.rotation.mT, pts.T)
    trans_loss = F.mse_loss(tru_pose.translation, pre_pose.translation)
    loss = F.mse_loss(tru_pts, pre_pts) + trans_loss
    loss = config["weight_loss"] * loss
    return loss

def sample_cube_points(n, size=100):
    coords = np.arange(-(n // 2), n // 2 + 1, dtype=np.float32) * (size / (n - 1))  # shape: (n,)
    assert coords[n // 2] == 0.0, "Center coordinate is not exactly 0!"
    grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing='ij')
    points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return torch.from_numpy(points)


def run_single_pose(standard_pose: str, volume_path: str, config: dict):
    """
    执行单个标准姿态（PA 或 RLAT）的完整推理流程，返回欧拉角。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取该姿态的配置
    pose_config = config['noise_params'][standard_pose]
    model_path = pose_config['model_path']
    trans_noise_range = torch.tensor(pose_config['trans_noise_range'])
    rota_noise_range = torch.tensor(pose_config['rota_noise_range'])

    # 1. 加载 CT
    subject = read(
        volume_path,
        bone_attenuation_multiplier=1.0,
        orientation=standard_pose,
        sid=500
    )

    # 2. 构建 DRR 渲染器
    drr = DRR(
        subject,
        sdd=800,
        height=512,
        delx=0.469,
        renderer="trilinear"
    ).to(device)

    # 3. 生成随机噪声位姿
    batch_size = config['batch_size']
    translation_noise = np.random.uniform(
        low=-np.array(trans_noise_range),
        high=np.array(trans_noise_range),
        size=(batch_size, 3)
    ).astype(np.float32)
    rotation_noise = np.random.uniform(
        low=-np.array(rota_noise_range),
        high=np.array(rota_noise_range),
        size=(batch_size, 3)
    ).astype(np.float32)

    rotations = torch.tensor(rotation_noise, dtype=torch.float32, device=device)
    translations = torch.tensor(translation_noise, dtype=torch.float32, device=device)
    # rotations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
    # translations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
    norm_rota_noise = rotations / np.array(rota_noise_range)
    norm_tran_noise = translations / np.array(trans_noise_range)
    label = torch.cat((
        torch.tensor(norm_rota_noise, dtype=torch.float32, device=device),
        torch.tensor(norm_tran_noise, dtype=torch.float32, device=device)
    ), dim=1)

    # 4. 渲染 DRR 并归一化
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY", degrees=True)
    img = norm_img(img)  # [B, 1, H, W]

    # 5. 加载模型并推理
    model = GridModel(
        model_config=config['model_config'],
        num_classes=6
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    label_transformer = LabelTransform(config['noise_params'])

    with torch.no_grad():
        outputs = model(img)
        pre_rota, pre_trans = label_transformer.label2real(outputs)
        tru_rota, tru_trans = label_transformer.label2real(label)
    # 6. 构造位姿并计算 extrinsic
    pre_pose = convert(pre_rota, pre_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
    tru_pose = convert(tru_rota, tru_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
    pre_extrinsic = (drr.detector.reorient.compose(pre_pose)).inverse()
    tru_extrinsic = (drr.detector.reorient.compose(tru_pose)).inverse()
    rt_ct2o_pre = pre_extrinsic.matrix.cpu().numpy().squeeze(0)  # [4, 4]
    rt_ct2o_tru = tru_extrinsic.matrix.cpu().numpy().squeeze(0)

    # # 7. 转为欧拉角（注意：rotation_to_euler_angles 需支持 4x4 矩阵）
    # euler_angles = rotation_to_euler_angles(rt_ct2o)  # 假设返回 [3,] 或 [1,3]
    return rt_ct2o_pre,  rt_ct2o_tru



global_config = init_config()
volume_path = r"../data/spine107_img.nii.gz"
euler_dev_list = []
euler_dev_tran_list = []
# 分别运行 PA 和 RLAT
for i in range(1000):
    rt_ct2o_sz_pre, rt_ct2o_sz_tru = run_single_pose("PA", volume_path, global_config)
    rt_ct2o_sc_pre, rt_ct2o_sc_tru = run_single_pose("RLAT", volume_path, global_config)
    euler_dev, euler_dev_tran = reg_rt_transform(rt_ct2o_sz_norm=rt_ct2o_sz_tru,
                                                 rt_ct2o_sc_norm=rt_ct2o_sc_tru,
                                                 rt_ct2o_sz=rt_ct2o_sz_pre,
                                                 rt_ct2o_sc=rt_ct2o_sc_pre)
    # 保存结果（假设返回的是 numpy array 或 list）
    euler_dev_list.append(euler_dev.copy())          # 防止后续修改影响
    euler_dev_tran_list.append(euler_dev_tran.copy())
    print(f"第{i}次迭代：euler_dev = {euler_dev}, euler_dev_tran = {euler_dev_tran}")
# 转为 NumPy 数组便于分析（可选）
euler_dev_array = np.array(euler_dev_list)          # shape: (N, 6)
euler_dev_tran_array = np.array(euler_dev_tran_list)  # shape: (N, 6)

# 计算样本方差（ddof=1）
var_euler_dev = np.var(euler_dev_array, axis=0, ddof=1)
var_euler_dev_tran = np.var(euler_dev_tran_array, axis=0, ddof=1)

# --- 保存路径 ---
save_dir = "calibration_results"
os.makedirs(save_dir, exist_ok=True)

# 1. 保存原始迭代数据（.npy 便于后续加载）
np.save(os.path.join(save_dir, "euler_dev_array.npy"), euler_dev_array)
np.save(os.path.join(save_dir, "euler_dev_tran_array.npy"), euler_dev_tran_array)

# 2. 保存方差结果（.npy + .csv）
np.save(os.path.join(save_dir, "var_euler_dev.npy"), var_euler_dev)
np.save(os.path.join(save_dir, "var_euler_dev_tran.npy"), var_euler_dev_tran)

# 3. 保存为 CSV（方便 Excel 或文本查看）
np.savetxt(os.path.join(save_dir, "euler_dev_array.csv"), euler_dev_array, delimiter=",", fmt="%.8f")
np.savetxt(os.path.join(save_dir, "euler_dev_tran_array.csv"), euler_dev_tran_array, delimiter=",", fmt="%.8f")

# 方差是 1D 数组，保存为单行 CSV
np.savetxt(os.path.join(save_dir, "var_euler_dev.csv"), var_euler_dev.reshape(1, -1), delimiter=",", fmt="%.8f")
np.savetxt(os.path.join(save_dir, "var_euler_dev_tran.csv"), var_euler_dev_tran.reshape(1, -1), delimiter=",", fmt="%.8f")

print(f"✅ 所有数据已保存至目录: {os.path.abspath(save_dir)}")








