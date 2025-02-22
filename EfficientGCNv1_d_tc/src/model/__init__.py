import math
from torch import nn

from . import layers
from .nets import EfficientGCN
from .nets2 import EfficientGCN_2
from .nets3 import EfficientGCN_3
from .nets4 import EfficientGCN_4
from .nets5 import EfficientGCN_5

from .activations import *


__activations = {
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'hswish': HardSwish(inplace=True),
    'swish': Swish(inplace=True),
}

def rescale_block(block_args, scale_args, scale_factor):
    channel_scaler = math.pow(scale_args[0], scale_factor)
    depth_scaler = math.pow(scale_args[1], scale_factor)
    new_block_args = []
    for [channel, stride, depth] in block_args:
        channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
        depth = int(round(depth * depth_scaler))
        new_block_args.append([channel, stride, depth])
    return new_block_args

def create(model_type, act_type, block_args, scale_args, **kwargs):
    kwargs.update({
        'act': __activations[act_type],
        'block_args': rescale_block(block_args, scale_args, int(model_type[-1])),
    })
    if "Y2" in model_type:
        return EfficientGCN_2(**kwargs)
    elif "Y3" in model_type:
        return EfficientGCN_3(**kwargs)
    elif "Y4" in model_type:
        return EfficientGCN_4(**kwargs)
    elif "Y5" in model_type:
        return EfficientGCN_5(**kwargs)
    else:
        return EfficientGCN(**kwargs)
