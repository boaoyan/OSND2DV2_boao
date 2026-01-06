from typing import Callable, Dict, Optional, Type

import torch.nn as nn
from timm.layers import create_conv2d, DropPath, make_divisible, LayerType, \
    get_norm_act_layer, SqueezeExcite

ModuleType = Type[nn.Module]
SELayerType = Optional[Callable[..., nn.Module]]  # 支持 partial 和 lambda

def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class DynamicDepthwiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            pw_kernel_size: int = 1,
            pw_act: bool = False,
            s2d: int = 0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            channel_attention_layer = None,
            edffn_layer = None,
            drop_path_rate: float = 0.,
    ):

        super(DynamicDepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(in_chs, sd_chs, kernel_size=2, stride=2, padding='same')
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = 'same' if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        groups = num_groups(group_size, in_chs)
        self.edffn_layer = edffn_layer(in_chs, 3, False) if edffn_layer else nn.Identity()
        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size,
            stride=stride,
            dilation=dilation, padding=dw_pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)
        if channel_attention_layer == SqueezeExcite:
            self.channel_attention_layer = channel_attention_layer(in_chs, act_layer=act_layer)
        elif channel_attention_layer == eca_layer:
            self.channel_attention_layer = channel_attention_layer()
        else:
            self.channel_attention_layer = nn.Identity()
        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.edffn_layer(x)
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.channel_attention_layer(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class DynamicInvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            exp_kernel_size: int = 1,
            pw_kernel_size: int = 1,
            s2d: int = 0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            channel_attention_layer=None,
            edffn_layer=None,
            conv_kwargs: Optional[Dict] = None,
            drop_path_rate: float = 0.,
    ):
        super(DynamicInvertedResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(in_chs, sd_chs, kernel_size=2, stride=2, padding='same')
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = 'same' if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)

        self.edffn_layer = edffn_layer(in_chs, 3, False) if edffn_layer else nn.Identity()
        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)
        if channel_attention_layer == SqueezeExcite:
            self.channel_attention_layer = channel_attention_layer(mid_chs, act_layer=act_layer)
        elif channel_attention_layer == eca_layer:
            self.channel_attention_layer = channel_attention_layer()
        else:
            self.channel_attention_layer = nn.Identity()
        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size,
            stride=stride,
            dilation=dilation, groups=groups, padding=dw_pad_type, **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, inplace=True)

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.edffn_layer(x)
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.channel_attention_layer(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x