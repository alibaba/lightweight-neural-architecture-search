# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os, sys, time, logging
import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class ComputeMadnasScore(metaclass=ABCMeta):
    def __init__(self, cfg, logger=None):
        self.init_std = cfg.score_init_std
        self.init_std_act = cfg.score_init_std_act
        self.batch_size = cfg.score_batch_size
        self.resolution = cfg.score_image_size
        self.in_ch = cfg.score_image_channel
        self.ratio_coef = cfg.score_multi_ratio
        self.budget_layers = cfg.budget_layers
        self.align_budget_layers = cfg.align_budget_layers
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger


    def ratio_score(self, stages_num, block_std_list):

        if stages_num!=len(self.ratio_coef):
            raise ValueError("the length of the stage_features_list (%d) must be equal to the length of ratio_coef (%d)"%(
                            stages_num, len(self.ratio_coef)))
        self.logger.debug("\nlen of stage_features_list:%d, len of block_std_list:%d\n%s"%(
                        stages_num, len(block_std_list), 
                        [std for std in block_std_list]))
        self.logger.debug("stage_idx:%s, stage_block_num:%s, stage_layer_num:%s\n"%(
                        self.stage_idx, self.stage_block_num, self.stage_layer_num))

        nas_score_list = []
        for idx, ratio in enumerate(self.ratio_coef):
            if ratio==0:
                nas_score_list.append(0.0)
                continue

            # compute std scaling
            nas_score_std = 0.0
            for idx1 in range(self.stage_block_num[idx]):
                nas_score_std += block_std_list[idx1]

            # larger channel and larger resolution, larger performance.
            resolution_stage = self.resolution//(2**(idx+1))
            nas_score_feat = np.log(self.stage_channels[idx]) 
            # different stage with the different feature map ratio (2**(idx+6))/(4**(idx+1))
            nas_score_stage = nas_score_std + nas_score_feat
            # layer_num_idx = self.stage_layer_num[idx]
            # if self.align_budget_layers: nas_score_stage = nas_score_stage/layer_num_idx*self.budget_layers
            self.logger.debug("stage:%d, nas_score_stage:%.3f, score_feat:%.3f, log_std:%.3f, resolution:%d"%(
                                idx, nas_score_stage, nas_score_feat, nas_score_std, resolution_stage))

            nas_score_list.append(nas_score_stage*ratio)
        self.logger.debug("nas_score:%s"%(np.sum(nas_score_list)))
        
        return nas_score_list


    def __call__(self, model):
        info = {}
        timer_start = time.time()
        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels = model.get_stage_info()
        kwarg = {"init_std":self.init_std, "init_std_act":self.init_std_act}
        
        block_std_list = model.madnas_forward_pre_GAP(**kwarg)
        nas_score_once = self.ratio_score(len(self.stage_idx), block_std_list)

        timer_end = time.time()
        nas_score_once = np.array(nas_score_once)
        avg_nas_score = np.sum(nas_score_once)

        if self.align_budget_layers: 
            nas_score_once = nas_score_once/self.stage_layer_num[-1]*self.budget_layers

        info['avg_nas_score'] = avg_nas_score
        info['std_nas_score'] = avg_nas_score
        info['nas_score_list'] = nas_score_once
        info['time'] = timer_end - timer_start
        self.logger.debug("avg_score:%s, consume time is %f ms\n"%(avg_nas_score, info['time']*1000))
        
        return info


def main():
    pass


if __name__ == '__main__':
    main()
    pass
