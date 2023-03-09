# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/mbv2_flops300e6/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 224  # 224 for Imagenet, 480 for detection, 160 for mcu

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info =[\
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 16, 's': 2, 'k': 3}, \
        {'class': 'SuperResK1DWK1', 'in': 16, 'out': 24, 's': 2, 'k': 3, 'L': 1, 'btn': 48}, \
        {'class': 'SuperResK1DWK1', 'in': 24, 'out': 48, 's': 2, 'k': 3, 'L': 1, 'btn': 96}, \
        {'class': 'SuperResK1DWK1', 'in': 48, 'out': 64, 's': 2, 'k': 3, 'L': 1, 'btn': 128}, \
        {'class': 'SuperResK1DWK1', 'in': 64, 'out': 96, 's': 1, 'k': 3, 'L': 1, 'btn': 192}, \
        {'class': 'SuperResK1DWK1', 'in': 96, 'out': 192, 's': 2, 'k': 3, 'L': 1, 'btn': 384}, \
        {'class': 'ConvKXBNRELU', 'in': 192, 'out': 1280, 's': 1, 'k': 1}, \
     ]
)

""" Budget config """
budgets = [
    dict(type = "flops", budget = 300e6),
    dict(type = "layers",budget = 53),
    ]

""" Score config """
score = dict(type = 'madnas', multi_block_ratio = [0,0,0,0,1])

""" Space config """
space = dict(
    type = 'space_k1dwk1',
    image_size = image_size,
    )

""" Search config """
search=dict(
    minor_mutation = False,  # whether fix the stage layer
    minor_iter = 100000,  # which iteration to enable minor_mutation
    popu_size = 256,
    num_random_nets = 100000,  # the searching iterations
    sync_size_ratio = 1.0,  # control each thread sync number: ratio * popu_size
    num_network = 1,
)

