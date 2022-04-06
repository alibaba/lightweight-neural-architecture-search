import os, sys
import random, copy, bisect


stem_mutate_method_list = ['out']
mutate_method_list = ['out', 'k', 'btn', 'L']
# last_block_mutate_method_list = ['btn', 'L']

the_maximum_channel = 1280
the_maximum_stem_channel = 32
search_kernel_size_list = [3, 5]
search_layer_list = [-2, -1, 1, 2]
search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5]
search_nbits_list = [2, 3, 4, 5, 6, 8]
nbits_mutate_ratio = 0.1 # random uniform 0~1, if bigger then mutate
search_btn_ratio_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


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


def mutate_function(block_id, structure_info_list, budget_layers, minor_mutation=False):

    structure_info = structure_info_list[block_id]
    #  Add the constraint: never change the last output channel
    if block_id == len(structure_info_list)-1:
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

        if random_mutate_method == 'L':
            old_L = copy.deepcopy(structure_info['L'])
            new_L = mutate_layer(structure_info['L'])

            # add the constraint: the block 1 can't have the large layers.
            if block_id==1:
                new_L = min(3, new_L)
            else:
                new_L = min(int(budget_layers//3//(len(structure_info_list)-3)), new_L)
            
            structure_info['L'] = new_L
        
        return [structure_info]
    
    else:
        raise RuntimeError('Not implemented class_name=' + class_name)