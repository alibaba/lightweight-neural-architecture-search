# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/R50_R480_FLOPs188e8_predictor/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 480  # 224 for Imagenet, 480 for detection, 160 for mcu

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = [ 
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 2, 'k': 3}, \
        {'class': 'SuperResK1KXK1', 'in': 32, 'out': 256, 's': 2, 'k': 3, 'L': 1, 'btn': 64}, \
        {'class': 'SuperResK1KXK1', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 1, 'btn': 128}, \
        {'class': 'SuperResK1KXK1', 'in': 512, 'out': 768, 's': 2, 'k': 3, 'L': 1, 'btn': 256}, \
        {'class': 'SuperResK1KXK1', 'in': 768, 'out': 1024, 's': 1, 'k': 3, 'L': 1, 'btn': 256}, \
        {'class': 'SuperResK1KXK1', 'in': 1024, 'out': 2048, 's': 2, 'k': 3, 'L': 1, 'btn': 512}, \
    ]
)

""" Latency config """
latency = dict(
    type = 'OpPredictor',
    data_type = "FP16",
    batch_size = 32,
    image_size = image_size,
    )

""" Budget config """
budgets = [
    dict(type = "flops", budget = 188e8),
    dict(type = "latency", budget = 8e-4),
    dict(type = "layers",budget = 91),
    ]

""" Score config """
score = dict(type = 'madnas', multi_block_ratio = [0,0,1,1,6])

""" Space config """
space = dict(
    type = 'space_k1kxk1',
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

