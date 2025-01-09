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


class SELayer(nn.Module):
    def __init__(self, inp, oup, act, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(oup, _make_divisible(inp // reduction, 8), kernel_size=1, stride=1, bias=False, padding=0),
            act,
            nn.Conv1d(_make_divisible(inp // reduction, 8), oup, kernel_size=1, stride=1, bias=False, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, _ = x.size()
        y = x.view(b * c, t, -1)
        y = self.avg_pool(y)
        y = y.view(b, c, t)
        y = self.conv(y).view(b, c, t, 1)
        return x * y


class SE_Temporal_Layer(nn.Module):
    def __init__(self, inp, oup, act, reduction=4, temporal_window_size=3):
        super(SE_Temporal_Layer, self).__init__()
        padding = (temporal_window_size - 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        inter_c = _make_divisible(inp // reduction, 8)
        self.conv = nn.Sequential(
            nn.Conv1d(oup, inter_c, kernel_size=1, stride=1, bias=False, padding=0),
            nn.Conv1d(inter_c, inter_c, kernel_size=temporal_window_size, stride=1, bias=False, padding=padding),
            act,
            nn.Conv1d(inter_c, oup, kernel_size=1, stride=1, bias=False, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, _ = x.size()
        y = x.view(b * c, t, -1)
        y = self.avg_pool(y)
        y = y.view(b, c, t)
        y = self.conv(y).view(b, c, t, 1)
        return x * y


class SE2_Temporal_Layer(nn.Module):
    def __init__(self, inp, oup, act, reduction=4, temporal_window_size=3):
        super(SE2_Temporal_Layer, self).__init__()
        padding = (temporal_window_size - 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        inter_c = _make_divisible(inp // reduction, 8)
        # self.conv = nn.Sequential(
        #     nn.Conv1d(oup, inter_c, kernel_size=1, stride=1, bias=False, padding=0),
        #     nn.Conv1d(inter_c, inter_c, kernel_size=temporal_window_size, stride=1, bias=False, padding=padding),
        #     act,
        #     nn.Conv1d(inter_c, oup, kernel_size=1, stride=1, bias=False, padding=0),
        #     nn.Sigmoid()
        # )
        self.conv = nn.Sequential(
            nn.Conv1d(oup, oup, kernel_size=temporal_window_size, stride=1, bias=False, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, _ = x.size()
        y = x.view(b * c, t, -1)
        y = self.avg_pool(y)
        y = y.view(b, c, t)
        y = self.conv(y).view(b, c, t, 1)
        return x * y


class SE_SpatialTemporal_Layer(nn.Module):
    def __init__(self, temporal_window_size):
        super(SE_SpatialTemporal_Layer, self).__init__()
        padding = (temporal_window_size - 1) // 2

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(temporal_window_size, temporal_window_size), stride=(1, 1),
                               bias=False, padding=(padding, padding))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, v = x.size()
        # N C T V --> N 1 T V -- N 1 T V
        y = x.mean(1, keepdim=True)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class SE2_SpatialTemporal_Layer(nn.Module):
    def __init__(self, inp, temporal_window_size, act, reduction=4):
        super(SE2_SpatialTemporal_Layer, self).__init__()
        padding = (temporal_window_size - 1) // 2
        inter_c = _make_divisible(inp // reduction, 8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, inter_c, kernel_size=1, stride=1, bias=False, padding=0),
            # nn.Conv1d(inter_c, inter_c, kernel_size=1, stride=1, bias=False, padding=padding),
            act,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(temporal_window_size, temporal_window_size), stride=(1, 1),
                      bias=False, padding=(padding, padding)),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, v = x.size()
        # N C T V --> N 1 T V -- N 1 T V
        y = self.conv1(x)
        y = y.mean(1, keepdim=True)
        y = self.conv2(y)
        return x * y


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


class Basic_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2d(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )


class Temporal_Bottleneck_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()

        inner_channel = channel // reduct_ratio
        padding = (temporal_window_size - 1) // 2
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.expand_conv = nn.Sequential(
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

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

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

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


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

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_SG_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size, 1), 1, (padding, 0), groups=channel, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=channel,
                      bias=bias),
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

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Temporal_MBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_MBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        hidden_dim = round(channel * expand_ratio)
        padding = (temporal_window_size - 1) // 2
        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(channel, hidden_dim, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=hidden_dim,
                      bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
            SELayer(channel, hidden_dim, self.act),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            y = self.conv(x)
            return x + y
        else:
            return self.conv(x)


class Temporal_FusedMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_FusedMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        hidden_dim = round(channel * expand_ratio)

        padding = (temporal_window_size - 1) // 2

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            # self.act,
        )
        # 与之前的版本不同的地方实质上是，conv1中没有以act结尾！———— 深度学习有些迷！
        self.conv2 = SELayer(channel, hidden_dim, self.act)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )
        params_add = sum(p.numel() for p in self.conv1.parameters() if p.requires_grad)
        print('conv1 params_add:{}| channel:{},hidden_dim:{},t:{},bias:{}'.format(params_add, channel, hidden_dim,
                                                                                  temporal_window_size, bias))
        params_add = sum(p.numel() for p in nn.BatchNorm2d(hidden_dim).parameters() if p.requires_grad)
        print('BN params_add:{}'.format(params_add))

        params_add = sum(p.numel() for p in self.conv2.parameters() if p.requires_grad)
        print('conv2 params_add:{}'.format(params_add))
        params_add = sum(p.numel() for p in self.conv3.parameters() if p.requires_grad)
        print('conv3 params_add:{}'.format(params_add))
        print("=================================================")

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce 1 x 1
class Temporal_STLiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_STLiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce T x 1
class Temporal_ST2LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_ST2LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# expand 1 x 1
class Temporal_ST3LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_ST3LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# expand T x 1
class Temporal_ST4LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_ST4LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce 1 x 1 + dw
class Temporal_ST5LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_ST5LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=hidden_dim,
                      bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# expand 1 x 1 + dw
class Temporal_ST6LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_ST6LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=hidden_dim,
                      bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce 1 x 1 + dw + act
class Temporal_ST7LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_ST7LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // reduct_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
            nn.Conv2d(hidden_dim, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=hidden_dim,
                      bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
        )
        self.conv2 = SE_SpatialTemporal_Layer(temporal_window_size)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce 1 x 1
class Temporal_TLiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_TLiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel

        hidden_dim = round(channel // expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act  # 这个很值得探究一下。
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce T x 1
class Temporal_T2LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_T2LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act  # 这个很值得探究一下。
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# expand 1 x 1
class Temporal_T3LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_T3LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act  # 这个很值得探究一下。
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# expand T x 1
class Temporal_T4LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_T4LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act  # 这个很值得探究一下。
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce 1 x 1 + dw
class Temporal_T5LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_T5LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,  # 这个很值得探究一下。
            nn.Conv2d(hidden_dim, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=hidden_dim,
                      bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# expand 1 x 1 + dw
class Temporal_T6LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_T6LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,  # 这个很值得探究一下。
            nn.Conv2d(hidden_dim, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=hidden_dim,
                      bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act,
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


# reduce T x 1 无 act
class Temporal_T7LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_T7LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            # self.act  # 这个很值得探究一下。
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


class Temporal_MultiLiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_MultiLiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel * expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (stride, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )

        self.se_st = SE_SpatialTemporal_Layer(temporal_window_size)
        self.se_t = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.se_c = SELayer(channel, hidden_dim, self.act)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.se_c(x) + x
            x_st = self.se_st(x) + x
            x_t = self.se_t(x) + x
            x = x_st + x_t + x
            x = self.conv3(x)
            return x + res
        else:
            # x = self.conv1(x)
            # x2 = self.conv2_2(x)
            # x = self.conv2(x)
            # x = self.conv3(x+x2)
            pass
            # return x


class Temporal_Multi2LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_Multi2LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )

        self.se_st = SE_SpatialTemporal_Layer(temporal_window_size)
        self.se_t = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.se_c = SELayer(channel, hidden_dim, self.act)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.se_c(x) + x
            x_st = self.se_st(x) + x
            x_t = self.se_t(x) + x
            x = x_st + x_t + x
            x = self.conv3(x)
            return x + res
        else:
            # x = self.conv1(x)
            # x2 = self.conv2_2(x)
            # x = self.conv2(x)
            # x = self.conv3(x+x2)
            pass
            # return x


class Temporal_Multi3LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_Multi3LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel
        padding = (temporal_window_size - 1) // 2

        hidden_dim = round(channel // expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
        )

        self.se_st = SE_SpatialTemporal_Layer(temporal_window_size)
        self.se_t = SE_Temporal_Layer(channel, hidden_dim, self.act, temporal_window_size=temporal_window_size)
        self.se_c = SELayer(channel, hidden_dim, self.act)

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.se_c(x)
            x_st = self.se_st(x)
            x_t = self.se_t(x)
            x = x_st + x_t + x
            x = self.conv3(x)
            return x + res
        else:
            # x = self.conv1(x)
            # x2 = self.conv2_2(x)
            # x = self.conv2(x)
            # x = self.conv3(x+x2)
            pass
            # return x


class Temporal_LiteMBConv_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio=4, stride=1, residual=True, **kwargs):
        super(Temporal_LiteMBConv_Layer, self).__init__()
        assert stride in [1, 2]
        oup = channel

        expand_ratio = 2
        hidden_dim = round(channel * expand_ratio)

        self.identity = stride == 1 and residual == True
        self.act = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, (1, 1), (1, 1), (0, 0), bias=bias),
            nn.BatchNorm2d(hidden_dim),
            self.act  # 这个很值得探究一下。
        )
        self.conv2 = SE_Temporal_Layer(channel, hidden_dim, self.act)
        # self.conv2 = SELayer(channel, hidden_dim, self.act)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            res = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x + res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x


class Temporal_Mta_Layer(nn.Module):
    expansion = 1

    def __init__(self, planes, temporal_window_size, act, bias, stride=1, scale=4, stype='normal',
                 foreconv=True, behindconv=True, **kwargs):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer. 作用于融合过程cons2
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.分支的数目。理论上scale*width == inplanes或planes、
            stype: 'normal': normal set. 'stage': first block of a new stage.
            如果stage，融合过程将不涉及级联融合
        """
        super(Temporal_Mta_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

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

        self.convs2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
                nn.BatchNorm2d(width),
                self.act
            ) for _ in range(self.nums)
        ])

        self.convs3 = nn.Sequential(
            nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion)
        ) if foreconv == True else nn.Identity()

        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )

        self.stype = stype
        self.scale = scale
        self.width = width
        self.foreconv = foreconv
        self.behindconv = behindconv

    def forward(self, x):
        residual = x

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
        out += residual

        return self.act(out)


class Temporal_MtaWrapper_Layer(nn.Module):
    expansion = 1

    def __init__(self, planes, temporal_window_size, act, bias, stride=1, scale=4, stype='normal',
                 foreconv=True, behindconv=True, layer_type="Sep", residual=True, ratio=2, **kwargs):
        """
        convtype: basic bottle  sep  epsep  sg
        """
        super(Temporal_MtaWrapper_Layer, self).__init__()
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

        # convtype = str(layer_type).split("-")[1]
        convtype = layer_type

        logging.info("Mta Layer Ratio: {}".format(self.ratio))
        stride = 1
        logging.info("Mta Layer stride: {}".format(stride))

        if convtype == "Basic":
            kernal_conv = Temporal_Basic_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                               act=self.act, residual=residual, stride=stride)
        elif convtype == "Bottleneck":
            kernal_conv = Temporal_Bottleneck_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                    bias=bias, act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                    stride=stride)
        elif convtype == "Sep":
            kernal_conv = Temporal_Sep_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                             act=self.act, residual=residual, stride=stride)
        elif convtype == "Epsep":
            kernal_conv = Temporal_Epsep_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                               act=self.act, expand_ratio=self.ratio, residual=residual, stride=stride)
        elif convtype == "SG":
            kernal_conv = Temporal_SG_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                            act=self.act, reduct_ratio=self.ratio, residual=residual, stride=stride)
        elif convtype == "MBConv":
            kernal_conv = Temporal_MBConv_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                                act=self.act, expand_ratio=self.ratio, residual=residual, stride=stride)
        elif convtype == "FusedMBConv":
            kernal_conv = Temporal_FusedMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                     bias=bias,
                                                     act=self.act, expand_ratio=self.ratio, residual=residual,
                                                     stride=stride)
        elif convtype == "LiteMBConv":
            kernal_conv = Temporal_LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                    bias=bias,
                                                    act=self.act, expand_ratio=self.ratio, residual=residual,
                                                    stride=stride)
        elif convtype == "STLiteMBConv":
            kernal_conv = Temporal_STLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "ST2LiteMBConv":
            kernal_conv = Temporal_ST2LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                       bias=bias,
                                                       act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                       stride=stride)
        elif convtype == "ST3LiteMBConv":
            kernal_conv = Temporal_ST3LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                       bias=bias,
                                                       act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                       stride=stride)
        elif convtype == "ST4LiteMBConv":
            kernal_conv = Temporal_ST4LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                       bias=bias,
                                                       act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                       stride=stride)
        elif convtype == "ST5LiteMBConv":
            kernal_conv = Temporal_ST5LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                       bias=bias,
                                                       act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                       stride=stride)
        elif convtype == "ST6LiteMBConv":
            kernal_conv = Temporal_ST6LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                       bias=bias,
                                                       act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                       stride=stride)
        elif convtype == "ST7LiteMBConv":
            kernal_conv = Temporal_ST7LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                       bias=bias,
                                                       act=self.act, reduct_ratio=self.ratio, residual=residual,
                                                       stride=stride)
        elif convtype == "TLiteMBConv":
            kernal_conv = Temporal_TLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                     bias=bias,
                                                     act=self.act, expand_ratio=self.ratio, residual=residual,
                                                     stride=stride)
        elif convtype == "T2LiteMBConv":
            kernal_conv = Temporal_T2LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, expand_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "T3LiteMBConv":
            kernal_conv = Temporal_T3LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, expand_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "T4LiteMBConv":
            kernal_conv = Temporal_T4LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, expand_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "T5LiteMBConv":
            kernal_conv = Temporal_T5LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, expand_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "T6LiteMBConv":
            kernal_conv = Temporal_T6LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, expand_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "T7LiteMBConv":
            kernal_conv = Temporal_T7LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                      bias=bias,
                                                      act=self.act, expand_ratio=self.ratio, residual=residual,
                                                      stride=stride)
        elif convtype == "MultiLiteMBConv":
            kernal_conv = Temporal_MultiLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                         bias=bias,
                                                         act=self.act, expand_ratio=self.ratio, residual=residual,
                                                         stride=stride)
        elif convtype == "MultiLiteMBConv":
            kernal_conv = Temporal_MultiLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                         bias=bias,
                                                         act=self.act, expand_ratio=self.ratio, residual=residual,
                                                         stride=stride)
        elif convtype == "Multi2LiteMBConv":
            kernal_conv = Temporal_Multi2LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                          bias=bias,
                                                          act=self.act, expand_ratio=self.ratio, residual=residual,
                                                          stride=stride)
        elif convtype == "Multi3LiteMBConv":
            kernal_conv = Temporal_Multi3LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                          bias=bias,
                                                          act=self.act, expand_ratio=self.ratio, residual=residual,
                                                          stride=stride)
        else:
            print("convtype:{}".format(convtype))
            raise ValueError("error convtype")

        self.convs2 = nn.ModuleList([kernal_conv for _ in range(self.nums)])

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
        params_add = sum(p.numel() for p in self.downsample.parameters() if p.requires_grad)
        print('self.downsample params_add:{}'.format(params_add))
        # params_add = sum(p.numel() for p in kernal_conv.parameters() if p.requires_grad)
        # print('kernal_conv params_add:{}'.format(params_add))
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

        if convtype == "Basic":
            self.convs2 = nn.ModuleList(
                [Temporal_Basic_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                      act=self.act, residual=residual, stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "Bottleneck":
            self.convs2 = nn.ModuleList(
                [Temporal_Bottleneck_Layer(channel=width, temporal_window_size=temporal_window_size,
                                           bias=bias, act=self.act, reduct_ratio=self.ratio, residual=residual,
                                           stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "Sep":
            self.convs2 = nn.ModuleList(
                [Temporal_Sep_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                    act=self.act, residual=residual, stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "Epsep":
            self.convs2 = nn.ModuleList(
                [Temporal_Epsep_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                      act=self.act, expand_ratio=self.ratio, residual=residual, stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "SG":
            self.convs2 = nn.ModuleList(
                [Temporal_SG_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                   act=self.act, reduct_ratio=self.ratio, residual=residual, stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "MBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_MBConv_Layer(channel=width, temporal_window_size=temporal_window_size, bias=bias,
                                       act=self.act, expand_ratio=self.ratio, residual=residual, stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "FusedMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_FusedMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                            bias=bias,
                                            act=self.act, expand_ratio=self.ratio, residual=residual,
                                            stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                           bias=bias,
                                           act=self.act, expand_ratio=self.ratio, residual=residual,
                                           stride=stride)
                 for _ in range(self.nums)])

        elif convtype == "STLiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_STLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, reduct_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "ST2LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_ST2LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                              bias=bias,
                                              act=self.act, reduct_ratio=self.ratio, residual=residual,
                                              stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "ST3LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_ST3LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                              bias=bias,
                                              act=self.act, reduct_ratio=self.ratio, residual=residual,
                                              stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "ST4LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_ST4LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                              bias=bias,
                                              act=self.act, reduct_ratio=self.ratio, residual=residual,
                                              stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "ST5LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_ST5LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                              bias=bias,
                                              act=self.act, reduct_ratio=self.ratio, residual=residual,
                                              stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "ST6LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_ST6LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                              bias=bias,
                                              act=self.act, reduct_ratio=self.ratio, residual=residual,
                                              stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "ST7LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_ST7LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                              bias=bias,
                                              act=self.act, reduct_ratio=self.ratio, residual=residual,
                                              stride=stride)
                 for _ in range(self.nums)])

        elif convtype == "TLiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_TLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                            bias=bias,
                                            act=self.act, expand_ratio=self.ratio, residual=residual,
                                            stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "T2LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_T2LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, expand_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "T3LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_T3LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, expand_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "T4LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_T4LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, expand_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "T5LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_T5LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, expand_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "T6LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_T6LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, expand_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "T7LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_T7LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                             bias=bias,
                                             act=self.act, expand_ratio=self.ratio, residual=residual,
                                             stride=stride)
                 for _ in range(self.nums)])

        elif convtype == "MultiLiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_MultiLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                bias=bias,
                                                act=self.act, expand_ratio=self.ratio, residual=residual,
                                                stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "MultiLiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_MultiLiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                bias=bias,
                                                act=self.act, expand_ratio=self.ratio, residual=residual,
                                                stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "Multi2LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_Multi2LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                 bias=bias,
                                                 act=self.act, expand_ratio=self.ratio, residual=residual,
                                                 stride=stride)
                 for _ in range(self.nums)])
        elif convtype == "Multi3LiteMBConv":
            self.convs2 = nn.ModuleList(
                [Temporal_Multi3LiteMBConv_Layer(channel=width, temporal_window_size=temporal_window_size,
                                                 bias=bias,
                                                 act=self.act, expand_ratio=self.ratio, residual=residual,
                                                 stride=stride)
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
        params_add = sum(p.numel() for p in self.downsample.parameters() if p.requires_grad)
        print('self.downsample params_add:{}'.format(params_add))
        # params_add = sum(p.numel() for p in kernal_conv.parameters() if p.requires_grad)
        # print('kernal_conv params_add:{}'.format(params_add))
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


# MS-G3D
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# MS-G3D
class Temporal_Ms_Layer(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2, 3, 4], residual=False,
                 residual_kernel_size=1, activation='relu', **kwargs):

        super(Temporal_Ms_Layer, self).__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride,
                             dilation=dilation))
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        return out


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )

# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel*self.s_kernel_size, 1, bias=bias)
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        return x


if __name__ == '__main__':
    # model = Basic_Layer(in_channel=32, out_channel=32, bias=True,
    #                     act=nn.ReLU(inplace=True), residual=True)
    # model_basic = Temporal_Basic_Layer(channel=32, temporal_window_size=3, bias=True,
    #                                    act=nn.ReLU(inplace=True), reduct_ratio=2)
    # model_bottle = Temporal_Bottleneck_Layer(channel=32, temporal_window_size=3, bias=True,
    #                                          act=nn.ReLU(inplace=True), reduct_ratio=2)
    # model_seq = Temporal_Sep_Layer(channel=32, temporal_window_size=3, bias=True,
    #                                act=nn.ReLU(inplace=True), expand_ratio=0)
    model_epseq1 = Temporal_FusedMBConv_Layer(channel=32, temporal_window_size=3, bias=True,
                                              act=nn.ReLU(inplace=True), expand_ratio=1)
    model_epseq2 = Temporal_FusedMBConv_Layer(channel=32, temporal_window_size=3, bias=True,
                                              act=nn.ReLU(inplace=True), expand_ratio=2)
    model_epseq4 = Temporal_FusedMBConv_Layer(channel=32, temporal_window_size=3, bias=True,
                                              act=nn.ReLU(inplace=True), expand_ratio=4)
    # model_sg = Temporal_SG_Layer(channel=32, temporal_window_size=3, bias=True,
    #                              act=nn.ReLU(inplace=True), reduct_ratio=2)

    # model_mta = Temporal_Mta_Layer(planes=32, scale=4, temporal_window_size=3, act=nn.ReLU(inplace=True), bias=True)
    # model_ms = Temporal_Ms_Layer(in_channels=32, out_channels=36)

    model_epseq1_mta = Temporal_MtaWrapper_Layer(planes=32, temporal_window_size=3, bias=True,
                                                 act=nn.ReLU(inplace=True), ratio=1,
                                                 layer_type="Epsep",
                                                 residual=True, stride=2)
    model_epseq2_mta = Temporal_MtaWrapper_Layer(planes=32, temporal_window_size=3, bias=True,
                                                 act=nn.ReLU(inplace=True), ratio=2,
                                                 layer_type="Epsep",
                                                 residual=True, stride=2)
    model_epseq4_mta = Temporal_MtaWrapper_Layer(planes=32, temporal_window_size=3, bias=True,
                                                 act=nn.ReLU(inplace=True), ratio=4,
                                                 layer_type="Epsep",
                                                 residual=True, stride=2)
    N, C, T, V = 2, 32, 50, 12
    data = torch.randn((N, C, T, V))
    # output = model(data)
    # print(output.shape)
    # output = model_basic(data)
    # print(output.shape)
    # output = model_bottle(data)
    # print(output.shape)
    # output = model_seq(data)
    # print(output.shape)
    output = model_epseq1(data)
    print(output.shape)
    output = model_epseq2(data)
    print(output.shape)
    output = model_epseq4(data)
    print(output.shape)
    # output = model_mta(data)
    # print(output.shape)
    # output = model_ms(data)
    # print(output.shape)

    print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    output = model_epseq1_mta(data)
    print(output.shape)
    output = model_epseq2_mta(data)
    print(output.shape)
    output = model_epseq4_mta(data)
    print(output.shape)

    print("=====================================")
    # params_add = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('model params_add:{}'.format(params_add))
    # params_add = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
    # print('model_basic params_add:{}'.format(params_add))
    # params_add = sum(p.numel() for p in model_bottle.parameters() if p.requires_grad)
    # print('model_bottle params_add:{}'.format(params_add))
    # params_add = sum(p.numel() for p in model_seq.parameters() if p.requires_grad)
    # print('model_seq params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq1.parameters() if p.requires_grad)
    print('model_epseq1 params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq2.parameters() if p.requires_grad)
    print('model_epseq2 params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq4.parameters() if p.requires_grad)
    print('model_epseq4 params_add:{}'.format(params_add))
    # params_add = sum(p.numel() for p in model_sg.parameters() if p.requires_grad)
    # print('model_sg params_add:{}'.format(params_add))
    # # --------------------------------------------
    # params_add = sum(p.numel() for p in model_mta.parameters() if p.requires_grad)
    # print('model_mta params_add:{}'.format(params_add))
    # params_add = sum(p.numel() for p in model_ms.parameters() if p.requires_grad)
    # print('model_ms params_add:{}'.format(params_add))

    params_add = sum(p.numel() for p in model_epseq1_mta.parameters() if p.requires_grad)
    print('model_epseq1_mta params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq2_mta.parameters() if p.requires_grad)
    print('model_epseq2_mta params_add:{}'.format(params_add))
    params_add = sum(p.numel() for p in model_epseq4_mta.parameters() if p.requires_grad)
    print('model_epseq4_mta params_add:{}'.format(params_add))
    """
    torch.Size([2, 32, 50, 12])
    =====================================
    model params_add:1120
    model_basic params_add:1120
    model_bottle params_add:1472
    model_seq params_add:4640
    model_sg params_add:1424
    """
