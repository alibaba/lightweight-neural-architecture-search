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

@MUTATORS.register_module(module_name = 'SuperRes3DK1DWK1')
class SuperRes3DK1DWK1Mutator():
    def __init__(self, 
                mutate_method_list,
                search_channel_list, 
                search_kernel_size_list, 
                search_layer_list,
                search_btn_ratio_list,
                the_maximum_channel,
                budget_layers,
                *args,
                **kwargs):
        self.budget_layers = budget_layers

        minor_mutation_list = ['out', 'btn']
        kwargs.update(dict(candidates = minor_mutation_list))
        self.minor_method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = mutate_method_list))
        self.method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = search_channel_list), the_maximum_channel= the_maximum_channel)
        self.channel_mutator = build_channel_mutator(kwargs) 

        kwargs = dict(candidates = search_kernel_size_list ) 
        self.kernel_mutator = build_kernel_mutator(kwargs) 

        kwargs = dict(candidates = search_layer_list) 
        self.layer_mutator = build_layer_mutator(kwargs) 

        kwargs = dict(candidates = search_btn_ratio_list) 
        self.btn_mutator = build_btn_mutator(kwargs) 

        self.search_btn_ratio_list = search_btn_ratio_list

    def check_btn(self,btn_ratio):

        if btn_ratio not in self.search_btn_ratio_list:
            return min(self.search_btn_ratio_list, key=lambda x: abs(x - btn_ratio))
        else:
            return btn_ratio

    def __call__(self, block_id, structure_info_list, minor_mutation = False, *args, **kwargs):
        structure_info = structure_info_list[block_id]
        if block_id == len(structure_info_list) - 1:
            return structure_info 
        if block_id < len(structure_info_list) - 1:
            structure_info_next = structure_info_list[block_id + 1]
        structure_info = copy.deepcopy(structure_info)

        # coarse2fine mutation flag, only mutate the channels' output
        random_mutate_method = self.minor_method_mutator() if minor_mutation else self.method_mutator() 

        if random_mutate_method == 'out':
            btn_ratio = structure_info['btn'] / structure_info['out']
            btn_ratio = self.check_btn(btn_ratio)
            new_out = self.channel_mutator(structure_info['out'])
            # Add the constraint: output_channel <= 4*input_channel

            new_out = max(structure_info['in'], new_out)
            if block_id < len(structure_info_list) - 1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            structure_info['btn'] = smart_round(new_out * btn_ratio)

        if random_mutate_method == 'k':
            new_k_list = self.kernel_mutator((structure_info['kt'], structure_info['k']))
            structure_info['kt'] = new_k_list[0]
            structure_info['k'] = new_k_list[1]

        if random_mutate_method == 'btn':
            btn_ratio = structure_info['btn'] / structure_info['out']
            btn_ratio = self.check_btn(btn_ratio)
            new_btn_ratio = self.btn_mutator(btn_ratio)
            new_btn = smart_round(structure_info['out'] * new_btn_ratio)
            structure_info['btn'] = new_btn

        if random_mutate_method == 'L':
            new_L = self.layer_mutator(structure_info['L'])
            # add the constraint: the block 1 can't have the large layers.
            # add the constraint: the block 1 can't have the large layers.
            if block_id == 1:
                new_L = min(3, new_L)
            elif self.budget_layers:
                new_L = min(
                    int(self.budget_layers // 3 // (len(structure_info_list) - 3)),
                    new_L)

            structure_info['L'] = new_L

        return structure_info
