# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

SEARCHERS = Registry('searchers')

def build_searcher(name: str = 'default_searcher', default_args: dict = None):
    """ build searcher given a searcher name

    Args:
        name (str, optional):  Searcher name, if None, default searcher
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    cfg = dict(type=name)
    return build_from_cfg(cfg, SEARCHERS, default_args=default_args)

