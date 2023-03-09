# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

BUDGETS= Registry('budgets')

def build_budget(cfg, default_args: dict = None):
    """ build budget given a budget name

    Args:
        name (str, optional):  Model name, if None, default budget
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    cfg['name'] = cfg['type']
    return build_from_cfg(cfg, BUDGETS, default_args=default_args)

