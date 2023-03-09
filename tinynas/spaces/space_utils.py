# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth + 1]), list(
        cand_tuple[depth + 1:2 * depth + 1]), cand_tuple[-1]


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


def __check_block_structure_info_list_valid__(block_structure_info_list, budget_layers = None):
    if len(block_structure_info_list) < 1:
        return False

    # check how many conv layers 
    layers = 0
    for block_structure_info in block_structure_info_list:
        if 'L' not in block_structure_info.keys():
            layers += 1
        elif block_structure_info['class'] in ['SuperResK1KX', 'SuperResKXKX']:
            layers += block_structure_info['L'] * 2
        else:
            layers += block_structure_info['L'] * 3

    if budget_layers is not None and layers > budget_layers:
        return False

    return True


def adjust_structures_inplace(block_structure_info_list, image_size):

    # adjust channels
    last_channels = None
    for i, block_structure_info in enumerate(block_structure_info_list):
        if last_channels is None:
            last_channels = block_structure_info['out']
            continue
        else:
            block_structure_info_list[i]['in'] = last_channels
            last_channels = block_structure_info['out']

    # adjust kernel size <= feature map / 1.5
    resolution = image_size
    for i, block_structure_info in enumerate(block_structure_info_list):
        stride = block_structure_info['s']
        kernel_size = block_structure_info['k']

        while kernel_size * 1.5 > resolution:
            kernel_size -= 2

        block_structure_info['k'] = kernel_size

        resolution /= stride

    return block_structure_info_list
