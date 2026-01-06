# Portions of this code are derived from the MIT-licensed work by:
# Vivek Gopalakrishnan et al.
# https://github.com/eigenvivek/DiffDRR
#
# Original paper:
# @inproceedings{gopalakrishnan2022fast,
#   title={Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging},
#   author={Gopalakrishnan, Vivek and Golland, Polina},
#   booktitle={Workshop on Clinical Image-Based Procedures},
#   pages={1--11},
#   year={2022},
#   organization={Springer}
# }

import numpy as np
import torch
from torchio import ScalarImage, Subject


def read(
    volume: str,  # CT volume
    orientation: str = "AP",  # Frame-of-reference change
    bone_attenuation_multiplier: float = 1.0,  # Scalar multiplier on density of high attenuation voxels
    center_volume: bool = True,  # Move the volume's isocenter to the world origin
    sid=600, # Source-to-Isocenter Distance
) -> Subject:
    """
    Read an image volume from a variety of formats, and optionally, any
    given labelmap for the volume. Converts volume to a RAS+ coordinate
    system and moves the volume isocenter to the world origin.
    """
    volume = ScalarImage(volume)

    # Convert the volume to density (inlined transform_hu_to_density)
    volume_data = volume.data.to(torch.float32)
    # air = torch.where(volume_data <= -800)
    # soft_tissue = torch.where((-800 < volume_data) & (volume_data <= 350))
    # bone = torch.where(350 < volume_data)
    density = volume_data
    # density = torch.empty_like(volume_data)
    # density[air] = volume_data[soft_tissue].min()
    # density[soft_tissue] = volume_data[soft_tissue]
    # density[bone] = volume_data[bone] * bone_attenuation_multiplier
    density -= density.min()
    density /= density.max()
    density = ScalarImage(tensor=density, affine=volume.affine)

    # Frame-of-reference change
    if orientation == "AP":
        # Rotates the C-arm about the x-axis by 90 degrees
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, -1, sid],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation == "PA":
        # Rotates the C-arm about the x-axis by 90 degrees
        # Reverses the direction of the y-axis
        reorient = torch.tensor(
            [
                [-1, 0, 0, 0],
                [0, 0, 1, -sid],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation == "RLAT":
        reorient = torch.tensor(
            [
                [0, 0, -1, sid],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation is None:
        # Identity transform
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unrecognized orientation {orientation}")

    # Package the subject
    subject = Subject(
        volume=volume,
        orientation=orientation,
        reorient=reorient,
        density=density,
    )

    # Move the subject's isocenter to the origin in world coordinates (inlined canonicalize)
    if center_volume:
        for image in subject.get_images(intensity_only=False):
            isocenter = image.get_center()
            Tinv = np.array(
                [
                    [1.0, 0.0, 0.0, -isocenter[0]],
                    [0.0, 1.0, 0.0, -isocenter[1]],
                    [0.0, 0.0, 1.0, -isocenter[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            image.affine = Tinv.dot(image.affine)

    return subject