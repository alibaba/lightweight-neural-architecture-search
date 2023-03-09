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

def revise_nbits_for_layers(old_L, new_L, structure_info):
    nbitsA = structure_info["nbitsA"][0]
    nbitsW = structure_info["nbitsW"][0]
    inner_layers = len(structure_info["nbitsA"])//old_L
    if inner_layers not in [2, 3]:
        raise ValueError("inner_layers must be 2 or 3 for current superblock, not %d"%(inner_layers))

    if old_L<new_L:
        extra_l = new_L-old_L
        structure_info['nbitsA'] += [nbitsA]*inner_layers*extra_l
        structure_info['nbitsW'] += [nbitsW]*inner_layers*extra_l
    else:
        structure_info['nbitsA'] = structure_info['nbitsA'][:new_L*inner_layers]
        structure_info['nbitsW'] = structure_info['nbitsW'][:new_L*inner_layers]

    return structure_info

@MUTATORS.register_module(module_name = 'SuperQuantResK1DWK1')
class SuperQuantResK1DWK1Mutator():
    def __init__(self, 
                mutate_method_list,
                search_channel_list, 
                search_kernel_size_list, 
                search_layer_list,
                search_nbits_list, 
                search_btn_ratio_list,
                the_maximum_channel,
                budget_layers,
                nbits_ratio,
                *args,
                **kwargs):
        self.budget_layers = budget_layers

        minor_mutation_list = ['out', 'btn']
        kwargs.update(dict(candidates = minor_mutation_list))
        self.minor_method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = mutate_method_list))
        self.method_mutator = build_mutator(default_args = kwargs)

        kwargs.update(dict(candidates = search_channel_list, the_maximum_channel = the_maximum_channel))
        self.channel_mutator = build_channel_mutator(kwargs) 

        kwargs = dict(candidates = search_kernel_size_list ) 
        self.kernel_mutator = build_kernel_mutator(kwargs) 

        kwargs = dict(candidates = search_layer_list) 
        self.layer_mutator = build_layer_mutator(kwargs) 

        kwargs = dict(candidates = search_nbits_list, nbits_ratio = nbits_ratio) 
        self.nbits_list_mutator = build_nbits_list_mutator(kwargs) 

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
            if "nbitsA" not in structure_info or "nbitsW" not in structure_info:
                raise NameError("structure_info must have nbitsA and nbitsW\n%s"%(structure_info))
            new_nbitsA = mutate_nbits(structure_info['nbitsA'])
            new_nbitsW = mutate_nbits(structure_info['nbitsW'])
            structure_info['nbitsA'] = new_nbitsA
            structure_info['nbitsW'] = new_nbitsW
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
            new_k = self.kernel_mutator(structure_info['k'])
            structure_info['k'] = new_k

        if random_mutate_method == 'btn':
            btn_ratio = structure_info['btn'] / structure_info['out']
            btn_ratio = self.check_btn(btn_ratio)
            new_btn_ratio = self.btn_mutator(btn_ratio)
            new_btn = smart_round(structure_info['out'] * new_btn_ratio)
            structure_info['btn'] = new_btn

        if random_mutate_method == 'nbits':
            if "nbitsA" not in structure_info or "nbitsW" not in structure_info:
                raise NameError("structure_info must have nbitsA and nbitsW\n%s"%(structure_info))
            new_nbitsA_list = self.nbits_list_mutator(structure_info['nbitsA'], structure_info['L'])
            new_nbitsW_list = self.nbits_list_mutator(structure_info['nbitsW'], structure_info['L'])
            structure_info['nbitsA'] = new_nbitsA_list
            structure_info['nbitsW'] = new_nbitsW_list

        if random_mutate_method == 'L':
            old_L = copy.deepcopy(structure_info['L'])
            new_L = self.layer_mutator(structure_info['L'])
            # add the constraint: the block 1 can't have the large layers.
            if block_id == 1:
                new_L = min(3, new_L)
            elif self.budget_layers:
                new_L = min(
                    int(self.budget_layers // 3 // (len(structure_info_list) - 3)),
                    new_L)

            if new_L!=old_L:
                structure_info = revise_nbits_for_layers(old_L, new_L, structure_info)

            structure_info['L'] = new_L

        return structure_info
