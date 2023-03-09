# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch


from .builder import SCORES, build_score

@SCORES.register_module(module_name = 'ensemble')
class ComputeEnsembleScore(metaclass=ABCMeta):

    def __init__(self, scores = None, ratio = 1, logger = None, **kwargs): 

        self.ratio = ratio 
        self.logger = logger or logging
        self.compute_scores = []
        for cfg in scores:
            default_args = {'logger': self.logger}
            
            compute_score = build_score(cfg, default_args =  default_args)
            self.compute_scores.append(compute_score)

    def __call__(self, model):
        info = {}
        info['avg_nas_score'] = 0 
        info['std_nas_score'] = 0 
        info['nas_score_list'] = 0 
        timer_start = time.time()

        for compute_score in self.compute_scores:
            score_info = compute_score(model)
            for k, v in info.items():
                info[k] += score_info[k] * self.ratio  
        timer_end = time.time()
        info['time'] = timer_end - timer_start 

        self.logger.debug('avg_score:%s, consume time is %f ms' %
                          (info['avg_nas_score'], info['time'] * 1000))

        return info


def main():
    pass


if __name__ == '__main__':
    main()
    pass
