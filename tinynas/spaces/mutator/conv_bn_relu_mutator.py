# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import bisect
import copy
import os
import random
import sys

from ..space_utils import smart_round
from .builder import * 

@MUTATORS.register_module(module_name = 'ConvKXBNRELU')
class ConvKXBNRELUMutator():
    def __init__(self, stem_mutate_method_list, 
                search_channel_list, 
                search_kernel_size_list, 
                the_maximum_channel, 
                the_maximum_stem_channel:int =32,
                the_minimum_stem_channel:int = None, 
                *args, **kwargs):

        kwargs.update(dict(candidates = stem_mutate_method_list))
        self.method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = search_channel_list, the_maximum_channel = the_maximum_channel))
        self.channel_mutator = build_channel_mutator(kwargs) 

        kwargs = dict(candidates = search_kernel_size_list ) 
        self.kernel_mutator = build_kernel_mutator(kwargs) 
        self.the_maximum_stem_channel = the_maximum_stem_channel
        self.the_minimum_stem_channel = the_minimum_stem_channel 


    def __call__(self, block_id, structure_info_list, *args, **kwargs):

        structure_info = structure_info_list[block_id]

        if block_id == len(structure_info_list) - 1:
            return structure_info 

        if block_id < len(structure_info_list) - 1:
            structure_info_next = structure_info_list[block_id + 1]
        structure_info = copy.deepcopy(structure_info)
        random_mutate_method = self.method_mutator()

        if random_mutate_method == 'out':
            new_out = self.channel_mutator(structure_info['out'])
            # Add the constraint: the maximum output of the stem block is 128
            new_out = min(self.the_maximum_stem_channel, new_out)
            if self.the_minimum_stem_channel:
                new_out = max(self.the_minimum_stem_channel, new_out)

            if block_id < len(structure_info_list) - 1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            return structure_info

        if random_mutate_method == 'k':
            new_k = self.kernel_mutator(structure_info['k'])
            structure_info['k'] = new_k
            return structure_info

