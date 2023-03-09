# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import numpy as np

from .base import CnnBaseSpace
from .builder import SPACES
from .mutator import build_mutator 

@SPACES.register_module(module_name = 'space_k1dwsek1')
class Spacek1dwsek1(CnnBaseSpace):  

    def __init__(self, name = None, 
                    image_size = 224, 
                    block_num = 2, 
                    exclude_stem = False, 
                    budget_layers=None, 
                    maximum_channel =1280, 
                    **kwargs): 

        super().__init__(name, image_size = image_size, 
                        block_num = block_num, 
                        exclude_stem = exclude_stem, 
                        budget_layers = budget_layers, 
                        **kwargs)

        kwargs = dict(stem_mutate_method_list = ['out'] ,
            the_maximum_channel = maximum_channel,
            mutate_method_list = ['out', 'k', 'btn', 'L'],
            search_kernel_size_list = [3, 5, 7],
            search_layer_list = [-2, -1, 1, 2],
            search_channel_list = np.arange(0.5, 2.01, 0.1),
            search_btn_ratio_list = np.arange(1.5, 6.01, 0.1),
            the_maximum_stem_channel = 32,
            the_minimum_stem_channel = 16,
            budget_layers = budget_layers
        )

        for n in ['ConvKXBNRELU','SuperResK1DWSEK1']:
            self.mutators[n] = build_mutator(n, kwargs)

