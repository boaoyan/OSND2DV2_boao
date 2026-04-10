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


def save_res():
    # Ensure output directory exists
    output_dir = "output"
    # Save the image
    filename = create_filename(rotations, translations)
    save_path = os.path.join(output_dir, filename)
    # Convert DRR image to numpy and save using matplotlib
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"图像已保存到: {save_path}")

if __name__ == '__main__':
    # Read in the volume and get its origin and spacing in world coordinates
    # subject = load_example_ct(orientation="AP")
    # volume_dir = "data/CT25/lum_25.nii.gz"
    volume_dir_2 = r"../data/spine107_img.nii.gz"
    subject = read(volume_dir_2, sid=500, orientation="RLAT")

    # Initialize the DRR module for generating synthetic X-rays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delx = 0.469
    height = 512
    drr = DRR(
        subject,     # An object storing the CT volume, origin, and voxel spacing
        sdd=800,  # Source-to-detector distance (i.e., focal length)
        height=height,  # Image height (if width is not provided, the generated DRR is square)
        delx=delx,    # Pixel spacing (in mm)
        renderer="trilinear"
    ).to(device)



    rotations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
    translations = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)

    # print(drr.detector.intrinsic)
    print("体素到世界")
    print(drr.affine.matrix)
    pose = convert(rotations, translations, parameterization="euler_angles", convention="ZXY", degrees=True)
    print("标准位姿RT")
    print(pose.matrix)
    # print(drr.detector.reorient.matrix)
    extrinsic = (drr.detector.reorient.compose(pose)).inverse()
    print("世界到光源")
    print(extrinsic.matrix)
    print("光源到世界")
    print(drr.detector.reorient.compose(pose).matrix)
    pt_in_plane = drr.perspective_projection(pose, torch.tensor([[[-100.0, 0.0, -100.0]]],
                                                                dtype=torch.float32, device=device))
    print("体素原点在像平面")
    print(pt_in_plane)

    unit = np.eye(4)
    unit_tensor = torch.tensor(unit, dtype=torch.float32, device=device)
    res = drr.affine_inverse(unit_tensor)
    print(res)

    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY", degrees=True)
    img = normalize_to_255(img)

    mask_img = apply_circular_mask(img)
    plot_drr(mask_img, ticks=False)
    # save_res()
    # save_tensor_as_image_pil(mask_img, "output/base.png")

    plt.show()