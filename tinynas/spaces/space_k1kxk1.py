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

@SPACES.register_module(module_name = 'space_k1kxk1')
class Spacek1kxk1(CnnBaseSpace):  
    
    def __init__(self, name = None, 
                    image_size = 224, 
                    block_num = 2, 
                    exclude_stem = False, 
                    budget_layers=None, 
                    maximum_channel =2048, 
                    **kwargs): 

        super().__init__(name, image_size = image_size, 
                        block_num = block_num, 
                        exclude_stem = exclude_stem, 
                        budget_layers = budget_layers, 
                        **kwargs)

        kwargs = dict(stem_mutate_method_list = ['out'] ,
            mutate_method_list = ['out', 'k', 'btn', 'L'],
            the_maximum_channel = maximum_channel,
            search_kernel_size_list = [3, 5], 
            search_layer_list = [-2, -1, 1, 2], 
            search_channel_list = [1.5, 1.25, 0.8, 0.6, 0.5],
            budget_layers = budget_layers,
            btn_minimum_ratio = 10  # the bottleneck must be larger than out/10
        )
        for n in ['ConvKXBNRELU','SuperResK1KXK1']:
            self.mutators[n] = build_mutator(n, kwargs)

