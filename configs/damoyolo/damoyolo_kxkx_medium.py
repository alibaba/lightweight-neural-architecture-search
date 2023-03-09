# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/kxkx_R640_layers20_lat4e-4/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 320  # 224 for Imagenet, 160 for mcu

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = [ 
        {'class': 'ConvKXBNRELU', 'in': 12, 'out': 32, 's': 1, 'k': 3}, 
        {'class': 'SuperResKXKX', 'in': 32, 'out': 48, 's': 2, 'k': 3, 'L': 1, 'btn': 32},
        {'class': 'SuperResKXKX', 'in': 48, 'out': 96, 's': 2, 'k': 3, 'L': 1, 'btn': 48}, 
        {'class': 'SuperResKXKX', 'in': 96, 'out': 96, 's': 2, 'k': 3, 'L': 1, 'btn': 96}, 
        {'class': 'SuperResKXKX', 'in': 96, 'out': 192, 's': 1, 'k': 3, 'L': 1, 'btn': 96}, 
        {'class': 'SuperResKXKX', 'in': 192, 'out': 384, 's': 2, 'k': 3, 'L': 1, 'btn': 192}, 
    ]
)

""" Latency config """
latency = dict(
    type = 'OpPredictor',
    data_type = "FP16_DAMOYOLO",
    batch_size = 32,
    image_size = image_size,
    )

""" Budget config """
budgets = [
    dict(type = "layers",budget = 20),
    dict(type = "latency", budget = 4e-4)
    ]

""" Score config """
score = dict(type = 'madnas', multi_block_ratio = [0,0,1,1,16])

""" Space config """
space = dict(
    type = 'space_kxkx',
    image_size = image_size,
    channel_range_list = [None, None, [64, 128], None, [128, 256], [256, 512]],
    kernel_size_list = [3],
    )

""" Search config """
search=dict(
    minor_mutation = False,  # whether fix the stage layer
    minor_iter = 100000,  # which iteration to enable minor_mutation
    popu_size = 512,
    num_random_nets = 500000,  # the searching iterations
    sync_size_ratio = 1.0,  # control each thread sync number: ratio * popu_size
    num_network = 1,
)

