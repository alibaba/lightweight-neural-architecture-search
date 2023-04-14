# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

work_dir = './save_model/E3DM_FLOPs_50e8/'
log_level = 'INFO'  # INFO/DEBUG/ERROR
log_freq = 1000

""" video config """
image_size = 224  
frames = 16

""" Model config """
model = dict(
    type = 'Cnn3DNet',
    structure_info = [ 
        {'class': 'Conv3DKXBNRELU', 'in': 3, 'out': 24, 's': 2, 'kt': 1, 'k': 3}, \
        {'class': 'SuperRes3DK1DWK1', 'in': 24, 'out': 24, 's': 2, 'kt': 1, 'k': 5, 'L': 1, 'btn': 48}, \
        {'class': 'SuperRes3DK1DWK1', 'in': 24, 'out': 48, 's': 2, 'kt': 3, 'k': 3, 'L': 1, 'btn': 96}, \
        {'class': 'SuperRes3DK1DWK1', 'in': 48, 'out': 96, 's': 2, 'kt': 3, 'k': 3, 'L': 1, 'btn': 192}, \
        {'class': 'SuperRes3DK1DWK1', 'in': 96, 'out': 96, 's': 1, 'kt': 3, 'k': 3, 'L': 1, 'btn': 192}, \
        {'class': 'SuperRes3DK1DWK1', 'in': 96, 'out': 192, 's': 2, 'kt': 3, 'k': 3, 'L': 1, 'btn': 384}, \
        {'class': 'Conv3DKXBNRELU', 'in': 192, 'out': 512, 's': 1, 'kt': 1, 'k': 1},\
    ]
)

""" Budget config """
budgets = [
    dict(type = "flops", budget = 50e8),
    dict(type = "layers",budget = 83),
    ]

""" Score config """
score = dict(type = 'stentr', multi_block_ratio = [0,0,0,0,1], frames=16)

""" Space config """
space = dict(
    type = 'space_3d_k1dwk1',
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

