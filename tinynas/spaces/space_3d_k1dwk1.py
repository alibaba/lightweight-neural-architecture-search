# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import bisect
import copy
import os
import random
import sys

from .base import CnnBaseSpace
from .builder import SPACES
from .mutator import build_mutator 


@SPACES.register_module(module_name = 'space_3d_k1dwk1')
class Space3Dk1dwk1(CnnBaseSpace):  

    def __init__(self, name = None, 
                    image_size = 224, 
                    block_num = 2, 
                    exclude_stem = False, 
                    budget_layers=None, 
                    maximum_channel =640, 
                    **kwargs): 

        super().__init__(name, image_size = image_size, 
                        block_num = block_num, 
                        exclude_stem = exclude_stem, 
                        budget_layers = budget_layers, 
                        **kwargs)

        kwargs = dict(stem_mutate_method_list = ['out'] ,
            the_maximum_channel = maximum_channel,
            mutate_method_list = ['out', 'k', 'btn', 'L'],
            search_kernel_size_list = [(1, 3), (1, 5), (3, 3)],
            search_layer_list = [-2, -1, 1, 2],
            search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5],
            search_btn_ratio_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            budget_layers = budget_layers
        )

        for n in ['Conv3DKXBNRELU','SuperRes3DK1DWK1']:
            self.mutators[n] = build_mutator(n, kwargs)

