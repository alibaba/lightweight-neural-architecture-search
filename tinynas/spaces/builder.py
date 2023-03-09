# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

SPACES = Registry('spaces')

def build_space(cfg, default_args: dict = None):
    """ build space given a space name

    Args:
        name (str, optional):  Space name, if None, default space
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(cfg, SPACES, default_args=default_args)

