import numpy as np
import torch
from registration.projector.drr import DRR
from registration.projector.read_data import read
batch_size = 2
rota_noise_range = [5, 5, 10]
trans_noise_range = [25, 50, 25]
volume_dir_2 = r"../data/spine107_img.nii.gz"
subject = read(volume_dir_2, bone_attenuation_multiplier=1.0, orientation="PA", sid=500)

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

# rotations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
# translations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
translation_noise = np.array([0, 0, 0], dtype=np.float32)
rotation_noise = np.array([0, 0, 0], dtype=np.float32)

# 按维度均匀分布生成随机数
# translation_noise = np.array(
#     [np.random.uniform(-r, r) for r in trans_noise_range], dtype=np.float32
# )
translation_noise = np.random.uniform(
    low=-np.array(trans_noise_range),
    high=np.array(trans_noise_range),
    size=(batch_size, 3)
).astype(np.float32)

# rotation_noise = np.array(
#     [np.random.uniform(-r, r) for r in rota_noise_range], dtype=np.float32
# )
rotation_noise = np.random.uniform(
    low=-np.array(rota_noise_range),
    high=np.array(rota_noise_range),
    size=(batch_size, 3)
).astype(np.float32)

rotations = torch.tensor(rotation_noise, dtype=torch.float32, device=device)
translations = torch.tensor(translation_noise, dtype=torch.float32, device=device)
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY", degrees=True)



print(img.shape)