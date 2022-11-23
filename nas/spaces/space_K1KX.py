# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os, sys
import random, copy, bisect


stem_mutate_method_list = ['out']
mutate_method_list = ['out', 'btn', 'L']
# last_block_mutate_method_list = ['btn', 'L']

# add channel limit range for 6 blocks (only valid for No. 3, 5, 6 layers)
channel_range = [None, None, [64, 128], None, [128, 256], [256, 512]]
the_maximum_channel = 2048
search_kernel_size_list = [3, 5]
search_layer_list = [-2, -1, 1, 2]
search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5]
btn_minimum_ratio = 10 # the bottleneck must be larger than out/10

def smart_round(x, base=8):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)


def mutate_channel(channels):
    scale = random.choice(search_channel_list)
    new_channels = smart_round(scale*channels)
    new_channels = min(the_maximum_channel, new_channels)
    return new_channels


def mutate_kernel_size(kernel_size):
    for i in range(len(search_kernel_size_list)):
        new_kernel_size = random.choice(search_kernel_size_list)
        if new_kernel_size!=kernel_size:
            break
    return new_kernel_size


def mutate_layer(layer):
    for i in range(len(search_layer_list)):
        new_layer = layer + random.choice(search_layer_list)
        new_layer = max(1, new_layer)
        if new_layer!=layer:
            break
    return new_layer


def mutate_function(block_id, structure_info_list, budget_layers, minor_mutation=False):

    structure_info = structure_info_list[block_id]
    if block_id < len(structure_info_list)-1: 
        structure_info_next = structure_info_list[block_id+1]
    structure_info = copy.deepcopy(structure_info)
    class_name = structure_info['class']

    if class_name == 'ConvKXBNRELU':
        if block_id <= len(structure_info_list) - 2:
            random_mutate_method = random.choice(stem_mutate_method_list)
        else:
            return False

        if random_mutate_method == 'out':
            new_out = mutate_channel(structure_info['out'])
            # Add the constraint: the maximum output of the stem block is 128
            new_out = min(32, new_out)
            if block_id < len(structure_info_list)-1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            return [structure_info]

        if random_mutate_method == 'k':
            new_k = mutate_kernel_size(structure_info['k'])
            structure_info['k'] = new_k
            return [structure_info]

    elif class_name == 'SuperResConvK1KX':
        # coarse2fine mutation flag, only mutate the channels' output
        mutate_method_list_final=['out', 'btn'] if minor_mutation else mutate_method_list

        random_mutate_method = random.choice(mutate_method_list_final)

        if random_mutate_method == 'out':
            new_out = mutate_channel(structure_info['out'])
            # Add the contraint: output_channel should be in range [min, max]
            if channel_range[block_id] is not None:
                this_min, this_max = channel_range[block_id]
                new_out = max(this_min, min(this_max, new_out))
            # Add the constraint: output_channel > input_channel
            new_out = max(structure_info['in'], new_out)
            if block_id < len(structure_info_list) - 1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            if block_id < len(structure_info_list) - 1 and "btn" in structure_info_next:
                new_btn = min(new_out, structure_info_next['btn'])
                structure_info_next['btn'] = new_btn

        if random_mutate_method == 'k':
            new_k = mutate_kernel_size(structure_info['k'])
            structure_info['k'] = new_k

        if random_mutate_method == 'btn':
            new_btn = mutate_channel(structure_info['btn'])
            # Add the constraint: bottleneck_channel <= output_channel
            new_btn = min(structure_info['out'], new_btn) 
            structure_info['btn'] = new_btn

        if random_mutate_method == 'L':
            new_L = mutate_layer(structure_info['L'])
            # add the constraint: the block 1 can't have the large layers.
            if block_id==1:
                new_L = min(2, new_L)
            else:
                new_L = min(int(budget_layers//2//(len(structure_info_list)-2)), new_L)

            structure_info['L'] = new_L

        # add the constraint: the btn must be larger than out/btn_minimum_ratio.
        if structure_info['btn']<(structure_info['out']/btn_minimum_ratio):
            structure_info['btn'] = smart_round(structure_info['out']/btn_minimum_ratio)
        
        return [structure_info]
    
    else:
        raise RuntimeError('Not implemented class_name=' + class_name)