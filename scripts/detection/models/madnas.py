# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
import torch
import torch.nn as nn
from models import MasterNet
from ..builder import BACKBONES

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module
class MadNas(nn.Module):
    def __init__(self, net_str=None, out_indices=(1, 2, 3, 4), init_cfg=None):
        super(MadNas, self).__init__()
        self.body = MasterNet(structure_txt=net_str, out_indices=out_indices, no_create=False)
        if init_cfg is not None and os.path.isfile(init_cfg):
            self.body.init_weights(init_cfg)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        """Forward function."""
        return self.body(x)


if __name__ == "__main__":
    pass