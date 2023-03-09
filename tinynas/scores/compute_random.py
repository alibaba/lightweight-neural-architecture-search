# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from random import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


from .builder import SCORES

@SCORES.register_module(module_name = 'random')
class ComputeRandomScore(metaclass=ABCMeta):

    def __init__(self, cfg, ratio = 1, logger=None, **kwargs):
        # TODO: to be finished after adding nas for transformer
        self.gpu = None
        self.ratio = ratio

        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

    def __call__(self, model):
        '''

        Args:
            model: Model to be compute scores

        Returns:
            A dictionary. Key 'avg_nas_score' is necessary. Others ['std_nas_score', 'nas_score_list', 'time'] are
            optional.

        '''
        model.eval()
        model.requires_grad_(False)

        info = {}
        nas_score_list = []
        timer_start = time.time()

        # TODO: calculate scores
        nas_score_list.append([random() * 100])

        timer_end = time.time()

        nas_score_list = np.array(nas_score_list) * self.ratio 
        avg_nas_score = np.mean(np.sum(nas_score_list, axis=1)) 
        std_nas_score = np.std(np.sum(nas_score_list, axis=1))

        info['avg_nas_score'] = avg_nas_score
        info['std_nas_score'] = std_nas_score
        info['nas_score_list'] = nas_score_list
        info['time'] = timer_end - timer_start
        self.logger.debug('avg_score:%s, consume time is %f ms\n' %
                          (avg_nas_score, info['time'] * 1000))

        del model
        torch.cuda.empty_cache()
        return info


def main():
    model = nn.Conv2d(3, 3, 3)
    score_compute = ComputeRandomScore(None)
    print(score_compute(model))


if __name__ == '__main__':
    main()
    pass
