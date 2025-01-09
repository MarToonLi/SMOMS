import logging

import torch
from torch import nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)


class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


class Temporal_Epsep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Epsep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )
        print("-----------------------------------------")
        params_add = sum(p.numel() for p in self.expand_conv.parameters() if p.requires_grad)
        print('self.expand_conv params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.depth_conv.parameters() if p.requires_grad)
        print('self.depth_conv params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.point_conv.parameters() if p.requires_grad)
        print('self.point_conv params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.residual.parameters() if p.requires_grad)
        print('self.downsample params_add:{}'.format(params_add))
        print("+++++++++++++++++++++++++++++++++++++++++")

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_Mta2Wrapper_Layer(nn.Module):
    expansion = 1

    def __init__(self, planes, temporal_window_size, act, bias, stride=1, scale=4, stype='normal',
                 foreconv=True, behindconv=True, layer_type="Sep", residual=True, ratio=2, **kwargs):
        """
        convtype: basic bottle  sep  epsep  sg
        """
        super(Temporal_Mta2Wrapper_Layer, self).__init__()
        self.act = act
        self.ratio = ratio

        assert planes % scale == 0
        width = int(planes // scale)

        self.convs1 = nn.Sequential(
            nn.Conv2d(planes, width * scale, kernel_size=1, bias=False),
            nn.BatchNorm2d(width * scale),
            self.act
        ) if foreconv == True else nn.Identity()

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convtype = layer_type
        # convtype = str(layer_type).split("-")[1]

        logging.info("Mta Layer Ratio: {}".format(self.ratio))
        stride = 1
        logging.info("Mta Layer stride: {}".format(stride))

        if convtype == "Epsep":
            self.convs2 = nn.ModuleList(
                [Temporal_Epsep_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                      act=self.act, expand_ratio=self.ratio, residual=residual, stride=stride)
                 for _ in range(self.nums)])
        else:
            print("convtype:{}".format(convtype))
            raise ValueError("error convtype")

        self.convs3 = nn.Sequential(
            nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        ) if foreconv == True else nn.Identity()

        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(planes, planes * self.expansion, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(planes * self.expansion),
            )

        self.stype = stype
        self.scale = scale
        self.width = width
        self.foreconv = foreconv
        self.behindconv = behindconv

        print("-----------------------------------------")
        params_add = sum(p.numel() for p in self.convs1.parameters() if p.requires_grad)
        print('self.convs1 params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.convs2.parameters() if p.requires_grad)
        print('self.convs2 params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.convs3.parameters() if p.requires_grad)
        print('self.convs3 params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.residual.parameters() if p.requires_grad)
        print('self.residual params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.downsample.parameters() if p.requires_grad)
        print('downsample params_add:{}'.format(params_add))
        print("+++++++++++++++++++++++++++++++++++++++++")

    def forward(self, x):
        res = self.residual(x)

        out = self.convs1(x)

        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs2[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            # 最后一个分支
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            # 最后一个分支经过pool
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.convs3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += res

        return self.act(out)


if __name__ == '__main__':
    model_epseq1 = Temporal_Epsep_Layer(channel=32, temporal_window_size=3, bias=True,
                                        act=nn.ReLU(inplace=True), expand_ratio=1)

    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    model_epseq2 = Temporal_Epsep_Layer(channel=32, temporal_window_size=3, bias=True,
                                        act=nn.ReLU(inplace=True), expand_ratio=2)
    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    model_epseq4 = Temporal_Epsep_Layer(channel=32, temporal_window_size=3, bias=True,
                                        act=nn.ReLU(inplace=True), expand_ratio=4)
    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    model_epseq1_mta = Temporal_Mta2Wrapper_Layer(planes=32, temporal_window_size=3, bias=True,
                                                  act=nn.ReLU(inplace=True), ratio=1,
                                                  layer_type="Epsep",
                                                  residual=True, stride=2)
    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    model_epseq2_mta = Temporal_Mta2Wrapper_Layer(planes=32, temporal_window_size=3, bias=True,
                                                  act=nn.ReLU(inplace=True), ratio=2,
                                                  layer_type="Epsep",
                                                  residual=True, stride=2)
    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    model_epseq4_mta = Temporal_Mta2Wrapper_Layer(planes=32, temporal_window_size=3, bias=True,
                                                  act=nn.ReLU(inplace=True), ratio=4,
                                                  layer_type="Epsep",
                                                  residual=True, stride=2)
    N, C, T, V = 2, 32, 50, 12
    data = torch.randn((N, C, T, V))

    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    params_add = sum(p.numel() for p in model_epseq1.parameters() if p.requires_grad)
    print('model_epseq1 params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq2.parameters() if p.requires_grad)
    print('model_epseq2 params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq4.parameters() if p.requires_grad)
    print('model_epseq4 params_add:{}'.format(params_add))

    params_add = sum(p.numel() for p in model_epseq1_mta.parameters() if p.requires_grad)
    print('model_epseq1_mta params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq2_mta.parameters() if p.requires_grad)
    print('model_epseq2_mta params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq4_mta.parameters() if p.requires_grad)
    print('model_epseq4_mta params_add:{}'.format(params_add))
