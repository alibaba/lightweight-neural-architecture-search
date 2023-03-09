# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from modelscope.utils.registry import Registry, build_from_cfg

LATENCIES = Registry('latencies')

def build_latency(cfg, default_args: dict = None):
    """ build latency given a latency name

    Args:
        name (str, optional):  Latency name, if None, default latency
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(cfg, LATENCIES, default_args=default_args)

