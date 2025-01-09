import logging

import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .layers_temporal import Spatial_Graph_Layer, Temporal_Basic_Layer


# I(STB)+M(SB) model (a)
class EfficientGCN_2(nn.Module):
    """
    基于EfficientGCN, 目的是将spatial block和spatial temporal block.的位置调换。
    """

    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EfficientGCN_2, self).__init__()

        num_input, num_channel, _, _, _ = data_shape
        self.block_args_2 = [[48, 1, 1],
                        [32, 1, 1],
                        [64, 2, 0],
                        [128, 2, 0],
                        ]

        # input branches
        self.input_branches = nn.ModuleList([EfficientGCN_Blocks(
            init_channel=stem_channel,
            block_args=self.block_args_2[:fusion_stage],
            input_channel=num_channel,
            **kwargs
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else self.block_args_2[fusion_stage - 1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel=num_input * last_channel,
            block_args=self.block_args_2[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = num_input * self.block_args_2[-1][0] if fusion_stage == len(self.block_args_2) else self.block_args_2[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        init_param(self.modules())

        params_add = sum(p.numel() for p in self.main_stream.parameters() if p.requires_grad)
        print('main_stream params_add:{}'.format(params_add))

    def forward(self, x):
        N, I, C, T, V, M = x.size()
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N * M, C, T, V)

        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.input_branches)], dim=1)

        x = self.main_stream(x)

        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        out = self.classifier(feature).view(N, -1)

        return out, feature


class EfficientGCN_Blocks(nn.Sequential):

    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()


        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        if "Mta" not in layer_type:
            temporal_layer = U.import_class(f'src.model.layers_temporal.Temporal_{layer_type}_Layer')
        else:
            block_type = layer_type.split("-")[0]
            sub_layer_type = layer_type.split("-")[1]
            kwargs["layer_type"] = sub_layer_type
            temporal_layer = U.import_class(f'src.model.layers_temporal.Temporal_{block_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_module(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = EfficientGCN(
        stem_channel=64,
        num_point=12,
        num_person=1,
        num_gcn_scales=13,
        num_g3d_scales=6,
        # graph='graph.ntu_rgb_d.AdjMatrixGraph',
        graph="graph.dad.AdjMatrixGraph"
    )

    N, C, T, V, M = 6, 3, 50, 12, 1
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))
