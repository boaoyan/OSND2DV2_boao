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

from typing import Optional
import torch
import torch.nn as nn
from .pose import RigidTransform, convert
from .detector import Detector
from .renderers import Siddon, Trilinear
from torchio import Subject


class DRR(nn.Module):
    """Differentiable Digitally Reconstructed Radiograph (DRR) generator.

    This module takes CT volume data and generates 2D X-ray projections (DRRs)
    from arbitrary viewpoints using differentiable rendering.
    """

    def __init__(
            self,
            subject: Subject,  # TorchIO wrapper for CT volume data
            sdd: float,  # Source-to-detector distance (mm)
            height: int,  # Output image height (pixels)
            delx: float,  # Pixel size in X direction (mm/pixel)
            width: Optional[int] = None,  # Output image width (defaults to height)
            dely: Optional[float] = None,  # Pixel size in Y direction (defaults to delx)
            x0: float = 0.0,  # Principal point X offset (pixels)
            y0: float = 0.0,  # Principal point Y offset (pixels)
            reshape: bool = True,  # Whether to reshape output to (batch, 1, height, width)
            reverse_x_axis: bool = False,  # Whether to flip X-axis for radiologic convention
            renderer: str = "siddon",  # Rendering backend ("siddon" or "trilinear")
            persistent: bool = True,  # Whether to persist buffers during state_dict()
            **renderer_kwargs,  # Additional renderer-specific parameters
    ):
        super().__init__()

        # Initialize detector geometry
        width = height if width is None else width
        dely = delx if dely is None else dely

        self.detector = Detector(
            sdd=sdd,
            height=height,
            width=width,
            delx=delx,
            dely=dely,
            x0=x0,
            y0=y0,
            reorient=subject.reorient,
            reverse_x_axis=reverse_x_axis,
        )

        # Store volume data and precompute transforms
        self.subject = subject
        self.reshape = reshape

        # Register persistent buffers for volume data and transforms
        self.register_buffer(
            "_affine",
            torch.as_tensor(subject.volume.affine, dtype=torch.float32).unsqueeze(0),
            persistent=persistent,
        )
        self.register_buffer(
            "_affine_inverse",
            self._affine.inverse(),
            persistent=persistent,
        )
        self.register_buffer(
            "density",
            subject.density.data.squeeze(),
            persistent=persistent,
        )

        # Initialize the selected renderer
        if renderer == "siddon":
            self.renderer = Siddon(**renderer_kwargs)
        elif renderer == "trilinear":
            self.renderer = Trilinear(**renderer_kwargs)
        else:
            raise ValueError(f"Unsupported renderer: {renderer}. Choose 'siddon' or 'trilinear'")

    @property
    def affine(self) -> RigidTransform:
        """Get the affine transform from voxel to world coordinates."""
        return RigidTransform(self._affine)

    @property
    def affine_inverse(self) -> RigidTransform:
        """Get the inverse affine transform from world to voxel coordinates."""
        return RigidTransform(self._affine_inverse)

    @property
    def device(self) -> torch.device:
        """Get the device where the volume data is stored."""
        return self.density.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the volume data."""
        return self.density.dtype

    def forward(
            self,
            *pose_args,  # Arguments representing the camera pose (SE(3))
            parameterization: Optional[str] = None,  # Rotation parameterization type
            convention: Optional[str] = None,  # Euler angle convention if applicable
            degrees=False,
            calibration: Optional[RigidTransform] = None,  # Detector intrinsic calibration
            **renderer_kwargs,  # Additional renderer parameters
    ) -> torch.Tensor:
        """Generate DRR from given camera pose.

        Args:
            pose_args: Camera pose parameters (interpretation depends on parameterization)
            parameterization: How rotation is represented (e.g., "euler", "quaternion")
            convention: Euler angle convention if using Euler angles
            degrees: Whether angles are in degrees (if applicable)
            calibration: Optional detector calibration transform
            renderer_kwargs: Additional parameters for the renderer

        Returns:
            Generated DRR image tensor
        """
        # Convert input pose arguments to standardized representation
        pose = (
            pose_args[0] if parameterization is None
            else convert(*pose_args, parameterization=parameterization, convention=convention, degrees=degrees)
        )

        # Compute source and target points in world coordinates
        source, target = self.detector(pose, calibration)

        # Initialize image with ray lengths (used as starting point for rendering)
        img = (target - source).norm(dim=-1).unsqueeze(1)

        # Transform rays from world to voxel coordinates
        source_voxel = self.affine_inverse(source)
        target_voxel = self.affine_inverse(target)

        # Perform the actual rendering
        img = self.renderer(
            self.density,
            source_voxel,
            target_voxel,
            img,
            **renderer_kwargs,
        )

        # Reshape output if requested (batch, 1, height, width)
        if self.reshape:
            img = img.view(
                len(pose),  # batch size
                -1,  # channels (1 for DRR)
                self.detector.height,
                self.detector.width,
            )

        return img

    def perspective_projection(
            self,
            pose: RigidTransform,
            pts: torch.Tensor,
    ):
        """Project points in world coordinates (3D) onto the pixel plane (2D)."""
        # Poses in DiffDRR are world2camera, but perspective transforms use camera2world, so invert
        extrinsic = (self.detector.reorient.compose(pose)).inverse()
        x = extrinsic(pts)

        # Project onto the detector plane
        x = torch.einsum("ij, bnj -> bni", self.detector.intrinsic, x)
        z = x[..., -1].unsqueeze(-1).clone()
        x = x / z

        # Move origin to upper-left corner
        x[..., 1] = self.detector.height - x[..., 1]
        if self.detector.reverse_x_axis:
            x[..., 0] = self.detector.width - x[..., 0]

        return x[..., :2]
