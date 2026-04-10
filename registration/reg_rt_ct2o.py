import torch

from registration.grid_efficientnet.grid_efficientb0_model import GridModel
from registration.label_transform import LabelTransform
from registration.projector.drr import DRR
from registration.projector.pose import convert
from registration.projector.post_processing import normalize_to_255
from registration.projector.read_data import read



def reg_ct2o(img, standard_pose, volume_path, model_path):
    """
    根据输入DRR图像和标准姿态，预测X射线投影外参矩阵。

    Args:
        img (torch.Tensor): 输入图像，shape [1, C, 224, 224]，已归一化
        standard_pose (str): "PA" 或 "RLAT"
        volume_path (str): CT volume 路径（如 "../data/spine107_img.nii.gz"）
        model_path (str): 预训练模型权重路径（.pth 文件）

    Returns:
        extrinsic (torch.Tensor): 4x4 外参矩阵，device 与 img 一致
    """
    assert standard_pose in ["PA", "RLAT"], f"Invalid pose: {standard_pose}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---- 0. 标准化输入图像 ----
    # img 是 (512, 512) 的 numpy array
    # print(img.shape)
    # img_tensor = torch.from_numpy(img)  # [1, 512, 512] → (C, H, W)
    # img= img_tensor  # [1, 1, 512, 512] → (B, C, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, 512, 512] → (C, H, W)
    img = img_tensor.unsqueeze(0)
    # print(img.shape)
    img = norm_img(img)

    # ---- 1. 加载 CT subject（每次调用都加载，避免全局状态）----
    subject = read(
        volume_path,
        bone_attenuation_multiplier=1.0,
        orientation=standard_pose,
        sid=500
    )

    # ---- 2. 构建 DRR 渲染器（固定参数）----
    delx = 0.469
    height = 512
    drr = DRR(
        subject,
        sdd=800,
        height=height,
        delx=delx,
        renderer="trilinear"
    ).to(device)

    # rotations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
    # translations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
    # img_new = drr(rotations, translations, parameterization="euler_angles", convention="ZXY", degrees=True)
    # img = normalize_to_255(img)
    # # save_tensor_as_image_pil(img, "data/ct_img/side_img_new2.png")
    # # np.save("data/ct_img/side_img_new2.npy", img_new)
    # print(img.shape)
    # img_new = norm_img(img_new)

    # # 判断是否在容差范围内相等（默认 atol=1e-08, rtol=1e-05）
    # are_close = torch.allclose(img, img_new, atol=1e-3, rtol=1e-3)
    # print("are_close", are_close)
    # ---- 3. 初始化 LabelTransform（使用与训练一致的噪声配置）----
    noise_params = {
        'standard_pose': standard_pose,
        'PA': {
            'trans_noise_range': [25, 25, 25],
            'rota_noise_range': [5, 5, 10],
        },
        'RLAT': {
            'trans_noise_range': [25, 25, 25],
            'rota_noise_range': [10, 5, 5],
        }
    }
    label_transformer = LabelTransform(noise_params)

    # ---- 4. 加载模型并推理 ----
    model = GridModel(
        model_config={'edffn': 1, 'eca': 0, 'conv_mlp': 0},
        num_classes=6
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(img)  # [1, 6]
        pre_rota, pre_trans = label_transformer.label2real(outputs)  # [1, 3], [1, 3]
        # outputs_new = model(img_new)
        # pre_rota_new, pre_trans_new = label_transformer.label2real(outputs_new)
    # print("Predicted rotation:", pre_rota)
    # print("Predicted trans:  ", pre_trans)
    # print("Predicted rotation new:", pre_rota_new)
    # print("Predicted trans new:  ", pre_trans_new)

    # ---- 5. 转换为 SE(3) 位姿并计算 extrinsic ----
    pose = convert(
        pre_rota,
        pre_trans,
        parameterization="euler_angles",
        convention="ZXY",
        degrees=True
    )

    # compose with detector reorientation and invert to get world-to-camera
    extrinsic = (drr.detector.reorient.compose(pose)).inverse()
    # print(type(extrinsic))
    # print(dir(extrinsic))
    rt_ct2o = extrinsic.matrix.cpu().numpy().squeeze(0)
    return rt_ct2o  # [4, 4] torch.Tensor

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