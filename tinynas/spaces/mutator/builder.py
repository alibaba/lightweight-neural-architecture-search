# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

MUTATORS= Registry('mutators')

def build_mutator(name = 'BaseMutator', default_args: dict = None):
    cfg = dict(type=name)
    return build_from_cfg(cfg, MUTATORS, default_args=default_args)

def build_channel_mutator(default_args: dict =None):
    return build_mutator('ChannelMutator', default_args = default_args)

def build_layer_mutator(default_args: dict =None):
    return build_mutator('LayerMutator', default_args = default_args)

def build_kernel_mutator(default_args: dict =None):
    return build_mutator('KernelMutator',  default_args = default_args)

def build_btn_mutator(default_args: dict =None):
    return build_mutator('BtnMutator', default_args =  default_args)

def build_nbits_list_mutator(default_args: dict =None):
    return build_mutator('NbitsListMutator', default_args =  default_args)
