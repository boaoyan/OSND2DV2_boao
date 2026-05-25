import pandas as pd
import torch

from projector.drr import DRR
from projector.pose import convert
from projector.post_processing import apply_circular_mask
from projector.read_data import read
from registration.metrics import *
from registration.registration import Registration


class RegisterInitHelper:
    def __init__(self, ct_config, init_pose_file_path=None, criterion="MSE"):
        volume_dir = ct_config.get("volume_dir")
        orientation = ct_config.get("train_pose")
        sid = ct_config.get("sid")
        sdd = ct_config.get("sdd")
        height = ct_config.get("im_sz")[0]
        delx = ct_config.get("delx")
        self.subject = read(volume_dir, orientation=orientation, sid=sid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.drr = DRR(
            self.subject,  # An object storing the CT volume, origin, and voxel spacing
            sdd=sdd,  # Source-to-detector distance (i.e., focal length)
            height=height,  # Image height (if width is not provided, the generated DRR is square)
            delx=delx,  # Pixel spacing (in mm)
            renderer="trilinear"
        ).to(self.device)
        self.delx = delx
        # 将真值设置为标准正位（PA）
        self.ground_rotations = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.ground_translations = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.ground_truth = self.drr(self.ground_rotations, self.ground_translations, parameterization="euler_angles", convention="ZXY")
        self.ground_truth = torch.max(self.ground_truth) - self.ground_truth
        self.ground_truth = apply_circular_mask(self.ground_truth)
        self.gt_pose = convert(self.ground_rotations, self.ground_translations, parameterization="euler_angles", convention="ZXY").to(self.device)
        # 起始位姿
        if init_pose_file_path is not None:
            self.init_pose_df = pd.read_csv(init_pose_file_path)
        else:
            print("未提供初始位姿文件")
        self.init_rotations, self.init_translations = None, None
        self.reg = None
        # 定义损失函数
        if criterion == "NCC":
            self.criterion = NormalizedCrossCorrelation2d()
        elif criterion == "MSE":
            self.criterion = MeanSquaredError2d()
        else:
            raise ValueError("未知的损失函数类型")
        # 使用多尺度NCC
        # self.criterion = MultiscaleNormalizedCrossCorrelation2d(
        #     patch_sizes=[3, 7, 15, 31, None],
        #     patch_weights=[0.1, 0.1, 0.2, 0.2, 0.4]
        # )
        # self.criterion = DoubleGeodesicSE3(800)

    def iter_initial_pose(self, iter_id):
        """
        根据给定的 iter_id 从预加载的 DataFrame 中读取初始位姿参数，
        构造旋转和平移张量。

        Args:
            iter_id (int): 要加载的初始位姿索引。
        """
        # 检查是否存在 init_pose_df
        if not hasattr(self, 'init_pose_df'):
            raise AttributeError("未找到 init_pose_df，请先加载初始位姿文件。")

        # 检查 iter_id 是否越界
        if iter_id >= len(self.init_pose_df) or iter_id < 0:
            raise IndexError(f"iter_id {iter_id} 越界，有效范围为 [0, {len(self.init_pose_df) - 1}]。")

        # 获取第 iter_id 行数据
        row = self.init_pose_df.iloc[iter_id]

        # 提取旋转和平移参数（单位应与训练/测试时一致）
        alpha = row['initial_rx']
        beta = row['initial_ry']
        gamma = row['initial_rz']
        bx = row['initial_tx']
        by = row['initial_ty']
        bz = row['initial_tz']

        # 构造旋转和平移张量，形状为 [1, 3]
        self.init_rotations = torch.tensor([[alpha, beta, gamma]], dtype=torch.float32, device=self.device)
        self.init_translations = torch.tensor([[bx, by, bz]], dtype=torch.float32, device=self.device)

    def set_initial_pose(self, data):
        alpha = data["alpha"]
        beta = data["beta"]
        gamma = data["gamma"]
        bx = data["bx"]
        by = data["by"]
        bz = data["bz"]
        self.init_rotations = torch.tensor([[alpha, beta, gamma]], dtype=torch.float32, device=self.device)
        self.init_translations = torch.tensor([[bx, by, bz]], dtype=torch.float32, device=self.device)

    def get_initial_pose(self):
        """返回当前的初始位姿 (rotations, translations)"""
        return self.init_rotations, self.init_translations

    def reinit(self):
        # if self.drr is not None:
        #     del self.drr
        # self.drr = DRR(
        #     self.subject,  # An object storing the CT volume, origin, and voxel spacing
        #     sdd=800,  # Source-to-detector distance (i.e., focal length)
        #     height=224,  # Image height (if width is not provided, the generated DRR is square)
        #     delx=self.delx,  # Pixel spacing (in mm)
        #     renderer="trilinear"
        # ).to(self.device)
        self.reg = Registration(
            self.drr,
            self.init_rotations.clone(),
            self.init_translations.clone(),
            parameterization="euler_angles",
            convention="ZXY",
        )

    def reverse_reinit(self):
        """
        将标准位置设置为初始位置
        :return:
        """
        self.reg = Registration(
            self.drr,
            self.ground_rotations.clone(),
            self.ground_translations.clone(),
            parameterization="euler_angles",
            convention="ZXY",
        )

    def get_optimizer_inputs(self):
        self.reinit()
        return {"reg": self.reg, "criterion": self.criterion, "ground_truth": self.ground_truth}
