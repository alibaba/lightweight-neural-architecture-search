# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import copy
import os
import sys
from .super_res_k1dwk1 import SuperResK1DWK1

class SuperQuantResK1DWK1(SuperResK1DWK1):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'SuperResK1DWK1',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
        }
        :param NAS_mode:
        '''
        super().__init__(
            structure_info=structure_info,
            no_create=no_create,
            dropout_channel=dropout_channel,
            dropout_layer=dropout_layer,
            **kwargs)


__module_blocks__ = {
    'SuperQuantResK1DWK1': SuperQuantResK1DWK1,
}
