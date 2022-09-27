# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os, sys
import random, copy, bisect


stem_mutate_method_list = ['out']
mutate_method_list = ['out', 'k', 'btn', 'L', 'nbits']
# last_block_mutate_method_list = ['btn', 'L']

the_maximum_channel = 1280
the_maximum_stem_channel = 32
search_kernel_size_list = [3, 5]
search_layer_list = [-2, -1, 1, 2]
search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5]
search_nbits_list = [3, 4, 5, 6]
nbits_mutate_ratio = 0.1 # random uniform 0~1, if bigger then mutate
search_btn_ratio_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


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


def mutate_btn_ratio(btn_ratio):
    for i in range(len(search_btn_ratio_list)):
        new_btn_ratio = random.choice(search_btn_ratio_list)
        if new_btn_ratio!=btn_ratio:
            break
    return new_btn_ratio


def mutate_layer(layer):
    for i in range(len(search_layer_list)):
        new_layer = layer + random.choice(search_layer_list)
        new_layer = max(1, new_layer)
        if new_layer!=layer:
            break
    return new_layer


def check_btn(btn_ratio):
    # new_btn_ratio = round(btn_ratio*2)/2
    # new_btn_ratio = max(new_btn_ratio, search_btn_ratio_list[0])
    # new_btn_ratio = min(new_btn_ratio, search_btn_ratio_list[-1])
    
    if btn_ratio not in search_btn_ratio_list:
        return min(search_btn_ratio_list, key=lambda x:abs(x-btn_ratio))
    else:
        return btn_ratio

def mutate_nbits(nbits):
    # avoid the endless loop
    if len(search_nbits_list)==1 and nbits in search_nbits_list:
        return nbits
    ind = search_nbits_list.index(nbits)
    new_ind = ind

    while new_ind == ind:
        new_ind = random.choice((ind - 1, ind + 1))
        new_ind = max(0, new_ind)
        new_ind = min(len(search_nbits_list) - 1, new_ind)
    return search_nbits_list[new_ind]


# def mutate_nbits_list(nbits_list):
#     if isinstance(nbits_list, int):
#         return mutate_nbits(nbits_list)
#     else:
#         # mutate 1/3 elements in nbits_list, the minimum number is 1
#         random_num = len(nbits_list)//3 if len(nbits_list)>=3 else 1
#         idx_list = random.sample(range(len(nbits_list)), random_num)
#         for idx in idx_list:
#             if random.uniform(0, 1)>nbits_mutate_ratio:
#                 nbits_list[idx] = mutate_nbits(nbits_list[idx])
#         return nbits_list


def mutate_nbits_list(nbits_list, L):
    if isinstance(nbits_list, int):
        return mutate_nbits(nbits_list)
    else:
        inner_layer = len(nbits_list)//L
        for layer_idx in range(L):
            if random.uniform(0, 1)>nbits_mutate_ratio:
                nbits_list[layer_idx*inner_layer:(layer_idx+1)*inner_layer] = \
                [mutate_nbits(nbits_list[layer_idx*inner_layer])]*inner_layer
        return nbits_list


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


def mutate_function(block_id, structure_info_list, budget_layers, minor_mutation=False):

    structure_info = structure_info_list[block_id]
    #  Add the constraint: never change the last output channel
    if block_id == len(structure_info_list)-1:
        if "nbitsA" not in structure_info or "nbitsW" not in structure_info:
            raise NameError("structure_info must have nbitsA and nbitsW\n%s"%(structure_info))
        new_nbitsA = mutate_nbits(structure_info['nbitsA'])
        new_nbitsW = mutate_nbits(structure_info['nbitsW'])
        structure_info['nbitsA'] = new_nbitsA
        structure_info['nbitsW'] = new_nbitsW
        return [structure_info]
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
            # Add the constraint: the maximum output of the stem block is 32
            new_out = min(the_maximum_stem_channel, new_out) 
            if block_id < len(structure_info_list)-1: 
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            return [structure_info]

        if random_mutate_method == 'k':
            new_k = mutate_kernel_size(structure_info['k'])
            structure_info['k'] = new_k
            return [structure_info]

    elif class_name == 'SuperResK1DWK1':
        # coarse2fine mutation flag, only mutate the channels' output
        mutate_method_list_final=['out', 'btn'] if minor_mutation else mutate_method_list

        random_mutate_method = random.choice(mutate_method_list_final)

        if random_mutate_method == 'out':
            btn_ratio = structure_info['btn']/structure_info['out']
            btn_ratio = check_btn(btn_ratio)
            new_out = mutate_channel(structure_info['out'])
            # Add the constraint: output_channel > input_channel
            new_out = max(structure_info['in'], new_out) 
            if block_id < len(structure_info_list)-1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            structure_info['btn'] = smart_round(new_out*btn_ratio)
                
        if random_mutate_method == 'k':
            new_k = mutate_kernel_size(structure_info['k'])
            structure_info['k'] = new_k

        if random_mutate_method == 'btn':
            btn_ratio = structure_info['btn']/structure_info['out']
            btn_ratio = check_btn(btn_ratio)
            new_btn_ratio = mutate_btn_ratio(btn_ratio)
            new_btn = smart_round(structure_info['out']*new_btn_ratio) 
            structure_info['btn'] = new_btn

        if random_mutate_method == 'nbits':
            if "nbitsA" not in structure_info or "nbitsW" not in structure_info:
                raise NameError("structure_info must have nbitsA and nbitsW\n%s"%(structure_info))
            new_nbitsA_list = mutate_nbits_list(structure_info['nbitsA'], structure_info['L'])
            new_nbitsW_list = mutate_nbits_list(structure_info['nbitsW'], structure_info['L'])
            structure_info['nbitsA'] = new_nbitsA_list
            structure_info['nbitsW'] = new_nbitsW_list

        if random_mutate_method == 'L':
            old_L = copy.deepcopy(structure_info['L'])
            new_L = mutate_layer(structure_info['L'])

            # add the constraint: the block 1 can't have the large layers.
            if block_id==1:
                new_L = min(3, new_L)
            else:
                new_L = min(int(budget_layers//3//(len(structure_info_list)-3)), new_L)
            
            if new_L!=old_L:
                structure_info = revise_nbits_for_layers(old_L, new_L, structure_info)

            structure_info['L'] = new_L
        
        return [structure_info]
    
    else:
        raise RuntimeError('Not implemented class_name=' + class_name)