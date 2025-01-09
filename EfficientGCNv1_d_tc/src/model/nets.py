import logging

import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .layers_temporal import Spatial_Graph_Layer, Temporal_Basic_Layer

# SMOMS
class EfficientGCN(nn.Module):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EfficientGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = nn.ModuleList([EfficientGCN_Blocks(
            init_channel=stem_channel,
            block_args=block_args[:fusion_stage],
            input_channel=num_channel,
            **kwargs
        ) for _ in range(num_input)])

        input_params_add = sum(p.numel() for p in self.input_branches.parameters() if p.requires_grad)
        print('1）input_branches params_add:{}'.format(input_params_add))

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage - 1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel=num_input * last_channel,
            block_args=block_args[fusion_stage:],
            **kwargs
        )

        main_params_add = sum(p.numel() for p in self.main_stream.parameters() if p.requires_grad)
        print('2） main_stream params_add:{}'.format(main_params_add))

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        classifier_params_add = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print('3） classifier params_add:{}'.format(classifier_params_add))

        init_param(self.modules())

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
            init_bn = nn.BatchNorm2d(input_channel)
            self.add_module('init_bn', init_bn)
            init_sgc = Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs)
            self.add_module('stem_scn', init_sgc)
            init_tc = Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs)
            self.add_module('stem_tcn', init_tc)

            init_bn_params_add = sum(p.numel() for p in init_bn.parameters() if p.requires_grad)
            print('1）a) init_bn params_add:{}'.format(init_bn_params_add))
            init_sgc_params_add = sum(p.numel() for p in init_sgc.parameters() if p.requires_grad)
            print('1）a) init_sgc params_add:{}'.format(init_sgc_params_add))
            init_tc_params_add = sum(p.numel() for p in init_tc.parameters() if p.requires_grad)
            print('1）a) init_tc params_add:{}'.format(init_tc_params_add))


        last_channel = init_channel
        if "Mta" not in layer_type:
            temporal_layer = U.import_class(f'src.model.layers_temporal.Temporal_{layer_type}_Layer')
        else:
            block_type = layer_type.split("-")[0]
            sub_layer_type = layer_type.split("-")[1]
            kwargs["layer_type"] = sub_layer_type
            temporal_layer = U.import_class(f'src.model.layers_temporal.Temporal_{block_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            block_sgc = Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs)
            self.add_module(f'block-{i}_scn', block_sgc)

            block_sgc_params = sum(p.numel() for p in block_sgc.parameters() if p.requires_grad)
            print('block-{}, block_sgc params_add:{}'.format(i, block_sgc_params))
            print("SGC related params: last_channel {}, channel {}, max_graph_distance {}.".format(
                last_channel, channel, max_graph_distance))

            for j in range(depth):
                s = stride if j == 0 else 1
                block_tc = temporal_layer(channel, temporal_window_size, stride=s, **kwargs)
                self.add_module(f'block-{i}_tcn-{j}', block_tc)

                block_tc_params = sum(p.numel() for p in block_tc.parameters() if p.requires_grad)
                print('block-{}_tcn-{},block_tc params_add:{}'.format(i,j,block_tc_params))

            block_att = Attention_Layer(channel, **kwargs)
            self.add_module(f'block-{i}_att', block_att)

            block_att_params = sum(p.numel() for p in block_att.parameters() if p.requires_grad)
            print('block-{} ,block_att params_add:{}'.format(i, block_att_params))

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
