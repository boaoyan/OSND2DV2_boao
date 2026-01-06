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


from typing import Optional, Tuple
import torch
from fastcore.basics import patch
from .pose import RigidTransform
# %% auto 0
__all__ = ['Detector', 'get_focal_length', 'get_principal_point', 'parse_intrinsic_matrix', 'make_intrinsic_matrix']

# %% ../notebooks/api/02_detector.ipynb 5


class Detector(torch.nn.Module):
    """
    A 6-DoF X-ray detector system model based on a C-Arm geometry.

    This class represents a radiographic detector system with configurable:
    - Intrinsic parameters (pixel spacing, principal point, source-to-detector distance)
    - Extrinsic parameters (pose/orientation in 3D space)

    Attributes:
        height (int): Detector height in pixels (Y-direction)
        width (int): Detector width in pixels (X-direction)
        reverse_x_axis (bool): Whether to reverse the X-axis (for handling reflections)
    """

    def __init__(
            self,
            sdd: float,
            height: int,
            width: int,
            delx: float,
            dely: float,
            x0: float,
            y0: float,
            reorient: torch.Tensor,
            reverse_x_axis: bool = False,
    ):
        """
        Initialize the detector system.

        Args:
            sdd: Source-to-detector distance (in length units)
            height: Detector height in pixels (Y-direction)
            width: Detector width in pixels (X-direction)
            delx: X-direction pixel spacing (length/pixel)
            dely: Y-direction pixel spacing (length/pixel)
            x0: Principal point x-coordinate (in length units)
            y0: Principal point y-coordinate (in length units)
            reorient: 4x4 frame-of-reference transformation matrix
            reverse_x_axis: Whether to reverse X-axis (default: False)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.reverse_x_axis = reverse_x_axis

        self._initialize_components(sdd, delx, dely, x0, y0, reorient)

    def _initialize_components(
            self,
            sdd: float,
            delx: float,
            dely: float,
            x0: float,
            y0: float,
            reorient: torch.Tensor,
    ) -> None:
        """Initialize the detector's core components."""
        # Initialize source and detector plane in default positions
        source, target = self._create_default_geometry()
        self.register_buffer("source", source)
        self.register_buffer("target", target)

        # Register reorientation transform
        self.register_buffer("_reorient", reorient)

        # Create calibration matrix
        self._create_calibration_matrix(sdd, delx, dely, x0, y0)

    def _create_calibration_matrix(
            self,
            sdd: float,
            delx: float,
            dely: float,
            x0: float,
            y0: float,
    ) -> None:
        """Create and register the 4x4 calibration matrix."""
        calibration_matrix = torch.tensor([
            [delx, 0, 0, x0],
            [0, dely, 0, y0],
            [0, 0, sdd, 0],
            [0, 0, 0, 1]
        ])
        self.register_buffer("_calibration", calibration_matrix)

    def _create_default_geometry(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create default source and detector plane positions.

        Returns:
            tuple: (source_position, target_positions)
                   source_position: (1, 1, 3) tensor at origin
                   target_positions: (1, N, 3) tensor of detector points
        """
        device = self._get_device()

        # Source at origin
        source = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 1, 3)

        # Detector plane centered at (0, 0, 1)
        target = self._create_detector_plane(device)

        return source, target

    def _create_detector_plane(self, device: torch.device) -> torch.Tensor:
        """
        Create coordinates for all detector pixels in default orientation.

        Args:
            device: Torch device for tensor creation

        Returns:
            (1, height*width, 3) tensor of detector coordinates
        """
        # Standard basis vectors for detector plane
        y_basis = torch.tensor([0.0, 1.0, 0.0], device=device)  # Y-axis (height)
        x_basis = torch.tensor([1.0, 0.0, 0.0], device=device)  # X-axis (width)

        # Create pixel coordinate offsets
        y_coords = self._create_pixel_coords(self.height)
        x_coords = self._create_pixel_coords(self.width)

        if not self.reverse_x_axis:
            x_coords = -x_coords

        # Create grid of pixel coordinates
        grid = torch.cartesian_prod(y_coords, x_coords)

        # Convert to 3D positions
        center = torch.tensor([0.0, 0.0, 1.0], device=device)
        target = torch.einsum('cd,nc->nd', torch.stack([y_basis, x_basis]), grid)
        target += center

        return target.unsqueeze(0)  # Add batch dimension

    def _create_pixel_coords(self, size: int) -> torch.Tensor:
        """
        Create 1D pixel coordinates with proper centering.

        Args:
            size: Number of pixels in one dimension

        Returns:
            1D tensor of pixel coordinates
        """
        offset = 1.0 if size % 2 else 0.5
        coords = torch.arange(-size // 2, size // 2) + offset
        return -coords.to(self._get_device())

    def _get_device(self) -> torch.device:
        """Get the device for tensor creation."""
        try:
            return self.sdd.device
        except AttributeError:
            return torch.device("cpu")

    @property
    def sdd(self) -> float:
        """Source-to-detector distance."""
        return self._calibration[2, 2].item()

    @property
    def delx(self) -> float:
        """X-direction pixel spacing (length/pixel)."""
        return self._calibration[0, 0].item()

    @property
    def dely(self) -> float:
        """Y-direction pixel spacing (length/pixel)."""
        return self._calibration[1, 1].item()

    @property
    def x0(self) -> float:
        """Principal point x-coordinate."""
        return -self._calibration[0, -1].item()

    @property
    def y0(self) -> float:
        """Principal point y-coordinate."""
        return -self._calibration[1, -1].item()

    @property
    def reorient(self) -> RigidTransform:
        """Get the reorientation transform as a RigidTransform."""
        return RigidTransform(self._reorient)

    @property
    def calibration(self) -> RigidTransform:
        """Get the calibration matrix as a RigidTransform."""
        return RigidTransform(self._calibration)

    @property
    def intrinsic(self) -> torch.Tensor:
        """Get the 3x3 intrinsic matrix."""
        return make_intrinsic_matrix(self).to(self.source)

    def forward(
            self,
            extrinsic: RigidTransform,
            calibration: Optional[RigidTransform] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute source and target points for X-ray projection.

        Args:
            extrinsic: RigidTransform representing the detector pose
            calibration: Optional alternate calibration (default: None)

        Returns:
            tuple: (source_points, target_points)
                   source_points: (1, 1, 3) tensor
                   target_points: (1, N, 3) tensor
        """
        # Apply calibration if provided, else use default
        target = calibration(self.target) if calibration else self.calibration(self.target)

        # Apply full pose transformation
        pose = self.reorient.compose(extrinsic)
        source = pose(self.source)
        target = pose(target)

        return source, target

# %% ../notebooks/api/02_detector.ipynb 9
def get_focal_length(
    intrinsic,  # Intrinsic matrix (3 x 3 tensor)
    delx: float,  # X-direction spacing (in units length)
    dely: float,  # Y-direction spacing (in units length)
) -> float:  # Focal length (in units length)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    return abs((fx * delx) + (fy * dely)).item() / 2.0

# %% ../notebooks/api/02_detector.ipynb 10
def get_principal_point(
    intrinsic,  # Intrinsic matrix (3 x 3 tensor)
    height: int,  # Y-direction length (in units pixels)
    width: int,  # X-direction length (in units pixels)
    delx: float,  # X-direction spacing (in units length)
    dely: float,  # Y-direction spacing (in units length)
):
    x0 = delx * (intrinsic[0, 2] - width / 2)
    y0 = dely * (intrinsic[1, 2] - height / 2)
    return x0.item(), y0.item()

# %% ../notebooks/api/02_detector.ipynb 11
def parse_intrinsic_matrix(
    intrinsic,  # Intrinsic matrix (3 x 3 tensor)
    height: int,  # Y-direction length (in units pixels)
    width: int,  # X-direction length (in units pixels)
    delx: float,  # X-direction spacing (in units length)
    dely: float,  # Y-direction spacing (in units length)
):
    focal_length = get_focal_length(intrinsic, delx, dely)
    x0, y0 = get_principal_point(intrinsic, height, width, delx, dely)
    return focal_length, x0, y0

# %% ../notebooks/api/02_detector.ipynb 12
def make_intrinsic_matrix(detector: Detector):
    fx = detector.sdd / detector.delx
    fy = detector.sdd / detector.dely
    u0 = detector.x0 / detector.delx + detector.width / 2
    v0 = detector.y0 / detector.dely + detector.height / 2
    return torch.tensor(
        [
            [fx, 0.0, u0],
            [0.0, fy, v0],
            [0.0, 0.0, 1.0],
        ]
    )
