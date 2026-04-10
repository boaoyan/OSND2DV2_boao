import numpy as np
import torch
import torch.nn.functional as F

from registration.grid_efficientnet.grid_efficientb0_model import GridModel
from registration.label_transform import LabelTransform
from registration.label_transform_mix import LabelTransformMix
from registration.projector.drr import DRR
from registration.projector.read_data import read
from registration.projector.pose import convert

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
            'trans_noise_range': [25, 25, 25],
            'rota_noise_range': [5, 5, 5]
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


global_config = init_config()
batch_size = global_config['batch_size']
rota_noise_range = torch.tensor(global_config['noise_params']['rota_noise_range'])
trans_noise_range = torch.tensor(global_config['noise_params']['trans_noise_range'])
label_transformer_mix = LabelTransformMix(global_config['noise_params'])
volume_dir_2 = r"../data/spine107_img.nii.gz"

subject = read(volume_dir_2, bone_attenuation_multiplier=1.0, orientation='RLAT', sid=500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


delx = 0.469
height = 512
drr = DRR(
    subject,  # An object storing the CT volume, origin, and voxel spacing
    sdd=800,  # Source-to-detector distance (i.e., focal length)
    height=height,  # Image height (if width is not provided, the generated DRR is square)
    delx=delx,  # Pixel spacing (in mm)
    renderer="trilinear"
).to(device)


model = GridModel(model_config=global_config['model_config'], num_classes=6).to(device)

# --- 生成噪声 ---
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

# --- 转为张量并生成 DRR ---
rotations = torch.tensor(rotation_noise, dtype=torch.float32, device=device)
translations = torch.tensor(translation_noise, dtype=torch.float32, device=device)
rotations = torch.tensor([[3, -3, 2]], dtype=torch.float32, device=device)
translations = torch.tensor([[5, -10, 20]], dtype=torch.float32, device=device)
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY", degrees=True)
print(img.shape)
img = norm_img(img)

# test_rot = torch.tensor([[3.0, 0.0, -2.0]], device=device)  # 在噪声范围内
# test_trans = torch.tensor([[10.0, 0.0, -15.0]], device=device)
#
# img_test = drr(test_rot, test_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
# img_test = norm_img(img_test)
# with torch.no_grad():
#     pred = model(img_test)
#     pred_rot, pred_trans = label_transformer_mix.label2real(pred)
# print(f"输入: rot={test_rot[0].tolist()}, trans={test_trans[0].tolist()}")
# print(f"预测: rot={pred_rot[0].tolist()}, trans={pred_trans[0].tolist()}")
# print(f"误差: rot={(pred_rot-test_rot).abs()[0].tolist()}, trans={(pred_trans-test_trans).abs()[0].tolist()}")

norm_rota_noise = rotations  / np.array(rota_noise_range)
norm_tran_noise = translations / np.array(trans_noise_range)
label = torch.cat((
    torch.tensor(norm_rota_noise, dtype=torch.float32, device=device),
    torch.tensor(norm_tran_noise, dtype=torch.float32, device=device)
), dim=1)
model.load_state_dict(torch.load(r"../data/reg_model/mix1_model.pth", map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    outputs = model(img)
    pre_rota, pre_trans = label_transformer_mix.label2real(outputs)
    tru_rota, tru_trans = label_transformer_mix.label2real(label)
    i=0
    print("Predicted rotation:", pre_rota[i])
    print("True rotation:    ", tru_rota[i])
    print("Predicted trans:  ", pre_trans[i])
    print("True trans:       ", tru_trans[i])
pose = convert(pre_rota, pre_trans, parameterization="euler_angles", convention="ZXY", degrees=True)
extrinsic = (drr.detector.reorient.compose(pose)).inverse()
rt_ct2o = extrinsic.matrix.cpu().numpy().squeeze(0)
print("世界到光源")
print(extrinsic.matrix)
print(rt_ct2o)