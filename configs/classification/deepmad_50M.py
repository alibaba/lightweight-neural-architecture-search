# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.
# The DeepMAD method is from the paper https://arxiv.org/abs/2303.02165 of Alibaba.

work_dir = './save_model/deepmad_50M/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" image config """
image_size = 224

""" Model config """
model = dict(
    type = 'CnnNet',
    structure_info = [ 
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 48, 's': 2, 'k': 3}, \
        {'class': 'SuperResK1DWSEK1', 'in': 48, 'out':84, 's': 2, 'k': 3, 'L': 1, 'btn': 504}, \
        {'class': 'SuperResK1DWSEK1', 'in': 84, 'out': 144, 's': 2, 'k': 3, 'L': 1, 'btn': 864}, \
        {'class': 'SuperResK1DWSEK1', 'in': 144, 'out': 216, 's': 2, 'k': 3, 'L': 1, 'btn': 1296}, \
        {'class': 'SuperResK1DWSEK1', 'in': 216, 'out': 324, 's': 2, 'k': 3, 'L': 1, 'btn': 1944}, \
        {'class': 'SuperResK1DWSEK1', 'in': 324, 'out': 512, 's': 1, 'k': 3, 'L': 1, 'btn': 3072}, \
        {'class': 'ConvKXBNRELU', 'in': 512, 'out': 2048, 's': 1, 'k': 1}, \
    ]
)

""" Budget config """
budgets = [
    dict(type = "flops", budget = 8.7e9),
    dict(type = "model_size", budget = 50e6),
    dict(type = "efficient_score", budget = 0.5)
    ]

""" Score config """
score = dict(type = 'deepmad', multi_block_ratio = [1,1,1,1,8], alpha1=1, alpha2=1, depth_penalty_ratio=10.)

""" Space config """
space = dict(
    type = 'space_k1dwsek1',
    image_size = image_size,
    )

""" Search config """
search=dict(
    minor_mutation = False,  # whether fix the stage layer
    minor_iter = 100000,  # which iteration to enable minor_mutation
    popu_size = 256,
    num_random_nets = 500000,  # the searching iterations
    sync_size_ratio = 1.0,  # control each thread sync number: ratio * popu_size
    num_network = 1,
)

