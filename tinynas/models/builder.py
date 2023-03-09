# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

MODELS= Registry('models')

def build_model(cfg, default_args: dict = None):
    """ build model given a model name

    Args:
        name (str, optional):  Model name, if None, default model
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(cfg, MODELS, default_args=default_args)

