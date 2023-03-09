# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

SCORES = Registry('scores')

def build_score(cfg, default_args: dict = None):
    """ build score given a score name

    Args:
        name (str, optional):  Score name, if None, default score
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(cfg, SCORES, default_args=default_args)

