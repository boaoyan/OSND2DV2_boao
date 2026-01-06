from itertools import product

import torch
import torch.nn as nn
from thop import profile
from timm.layers import SelectAdaptivePool2d, GroupNormAct, SqueezeExcite, ConvMlp
from functools import partial

from .edffn import EDFFN
from .dynamic_blocks import DynamicDepthwiseSeparableConv, DynamicInvertedResidual, eca_layer


class GridModel(nn.Module):
    def __init__(self, model_config, num_classes=6, in_channel=1):
        super().__init__()
        edffn_layer = EDFFN if model_config['edffn'] else None
        channel_attention_layer = eca_layer if model_config['eca'] else SqueezeExcite
        act_layer = nn.Tanh
        norm_layer = partial(GroupNormAct, group_size=8, act_layer=act_layer)
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            GroupNormAct(32, group_size=8, act_layer=act_layer),
        )

        # Stages
        layers = [
            # Stage 0 112*112
            DynamicDepthwiseSeparableConv(32, 16, stride=1,
                                   act_layer=act_layer, norm_layer=norm_layer,
                                          edffn_layer=edffn_layer, channel_attention_layer=channel_attention_layer),

            # Stage 1 56*56
            DynamicInvertedResidual(16, 24, stride=2, exp_ratio=6, norm_layer=norm_layer,
                             act_layer=act_layer, edffn_layer=edffn_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(24, 24, stride=1, exp_ratio=6, norm_layer=norm_layer,
                             act_layer=act_layer, edffn_layer=edffn_layer, channel_attention_layer=channel_attention_layer),

            # Stage 2 28*28
            DynamicInvertedResidual(24, 40, stride=2, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(40, 40, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),

            # Stage 3 14*14
            DynamicInvertedResidual(40, 80, stride=2, exp_ratio=6, dw_kernel_size=3, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(80, 80, stride=1, exp_ratio=6, dw_kernel_size=3, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(80, 80, stride=1, exp_ratio=6, dw_kernel_size=3, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),

            # Stage 4 14*14
            DynamicInvertedResidual(80, 112, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(112, 112, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(112, 112, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),

            # Stage 5 7*7
            DynamicInvertedResidual(112, 192, stride=2, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(192, 192, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(192, 192, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
            DynamicInvertedResidual(192, 192, stride=1, exp_ratio=6, dw_kernel_size=5, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),

            # Stage 6 7*7
            DynamicInvertedResidual(192, 320, stride=1, exp_ratio=6, dw_kernel_size=3, norm_layer=norm_layer,
                             act_layer=act_layer, channel_attention_layer=channel_attention_layer),
        ]
        in_chs = 320

        self.blocks = nn.Sequential(*layers)

        # Head
        if model_config["conv_mlp"]:
            self.head = nn.Sequential(
                ConvMlp(
                    in_features=in_chs,  # 匹配 Backbone 输出维度
                    hidden_features=512,  # 隐藏层维度
                    out_features=6,  # 输出6维向量
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ),
                SelectAdaptivePool2d(pool_type='avg', flatten=True)
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(in_chs, 1280, kernel_size=1, stride=1, bias=False),
                GroupNormAct(1280, group_size=8, act_layer=act_layer),
                SelectAdaptivePool2d(pool_type='avg', flatten=True),
                nn.Linear(1280, num_classes),
            )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    # 定义所有布尔组合
    keys = ['edffn', 'eca', 'conv_mlp']
    combinations = list(product([True, False], repeat=3))

    # 汇总结果
    results = []

    # 输入张量（1通道，224x224）
    x = torch.randn(1, 1, 224, 224)

    print(f"{'Config':<30} {'Params(M)':<12} {'FLOPs(G)':<10} {'Output Shape'}")
    print("-" * 60)

    for combo in combinations:
        model_config = dict(zip(keys, combo))
        # 构建模型
        model = GridModel(model_config, num_classes=6, in_channel=1)
        model.eval()

        # 参数统计
        params = sum(p.numel() for p in model.parameters())

        # FLOPs 统计
        flops, _ = profile(model, inputs=(x,), verbose=False)

        # 前向传播
        with torch.no_grad():
            out = model(x)

        # 格式化配置名
        config_str = f"edffn={combo[0]}, eca={combo[1]}, mlp={combo[2]}"
        results.append({
            'config': config_str,
            'params': params,
            'flops': flops,
            'output_shape': out.shape
        })

        # 打印当前行
        print(f"{config_str:<30} {params / 1e6:.2f}M       {flops / 1e9:.2f}G       {tuple(out.shape)}")

    print("-" * 60)