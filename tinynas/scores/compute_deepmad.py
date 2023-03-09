# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.
# The DeepMAD score is from the paper https://arxiv.org/abs/2303.02165 of Alibaba.

import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch


from .builder import SCORES

@SCORES.register_module(module_name = 'deepmad')
class ComputeDeepMadScore(metaclass=ABCMeta):

    def __init__(self, image_size = 224, multi_block_ratio = [1,1,1,1,8], ratio = 1, init_std = 1, init_std_act = 1, logger=None, **kwargs):
        self.init_std = init_std
        self.init_std_act = init_std_act
        self.resolution = image_size
        self.ratio_coef = multi_block_ratio
        self.ratio = ratio
        self.alpha1 = kwargs.get('alpha1', 1)
        self.alpha2 = kwargs.get('alpha2', 1)

        self.logger = logger or logging

    def ratio_score(self, stages_num, block_std_list):

        if stages_num != len(self.ratio_coef):
            raise ValueError(
                'the length of the stage_features_list (%d) must be equal to the length of ratio_coef (%d)'
                % (stages_num, len(self.ratio_coef)))
        self.logger.debug(
            'len of stage_features_list:%d, len of block_std_list:%d %s' %
            (stages_num, len(block_std_list), [std for std in block_std_list]))
        self.logger.debug(
            'stage_idx:%s, stage_block_num:%s, stage_layer_num:%s' %
            (self.stage_idx, self.stage_block_num, self.stage_layer_num))

        nas_score_list = []
        for idx, ratio in enumerate(self.ratio_coef):
            if ratio == 0:
                nas_score_list.append(0.0)
                continue

            # compute std scaling
            nas_score_std = 0.0
            for idx1 in range(self.stage_block_num[idx]):
                nas_score_std += block_std_list[idx1]

            # larger channel and larger resolution, larger performance.
            resolution_stage = self.resolution // (2**(idx + 1))

            nas_score_feat = np.log(self.stage_channels[idx])
            nas_score_stage = nas_score_feat
            nas_score_stage = nas_score_stage + np.log(self.stage_feature_map_size[idx] ** 2)

            nas_score_stage = nas_score_stage * nas_score_std

            self.logger.debug("stage:%d, mad_nas_score_stage:%.3f, score_feat:%.3f, log_std:%.3f, resolution:%d"%(
                                idx, nas_score_stage, nas_score_feat, nas_score_std, resolution_stage))
            nas_score_list.append(nas_score_stage * ratio)
        self.logger.debug("nas_score:%s"%(np.sum(nas_score_list)))

        return nas_score_list

    def __call__(self, model):
        info = {}
        timer_start = time.time()
        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels, self.stage_feature_map_size = model.get_stage_info(self.resolution)

        kwarg = {"alpha1":self.alpha1, "alpha2":self.alpha2}
        block_std_list = model.deepmad_forward_pre_GAP(**kwarg)
        nas_score_once = self.ratio_score(len(self.stage_idx), block_std_list)
        timer_end = time.time()

        nas_score_once = np.array(nas_score_once) * self.ratio 
        avg_nas_score = np.sum(nas_score_once) 

        info['avg_nas_score'] = avg_nas_score 
        info['std_nas_score'] = avg_nas_score
        info['nas_score_list'] = nas_score_once 
        info['time'] = timer_end - timer_start
        self.logger.debug('avg_score:%s, consume time is %f ms' %
                          (avg_nas_score, info['time'] * 1000))

        return info


def main():
    pass


if __name__ == '__main__':
    main()
    pass
