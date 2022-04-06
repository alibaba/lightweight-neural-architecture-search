# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os, sys, time, logging
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from abc import ABCMeta, abstractmethod


def network_weight_gaussian_init(net: nn.Module, std=1):
    with torch.no_grad():
        for m in net.modules():
            # import pdb; pdb.set_trace()
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=std)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.track_running_stats = True
                m.eps = 1e-5
                m.momentum = 1.0
                m.train()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net   


class ComputeEntropyScore(metaclass=ABCMeta):
    def __init__(self, cfg, logger=None):
        self.gpu = cfg.gpu
        self.init_std = cfg.score_init_std
        self.batch_size = cfg.score_batch_size
        self.resolution = cfg.score_image_size
        self.in_ch = cfg.score_image_channel
        self.repeat = cfg.score_repeat
        self.skip_relu = cfg.score_skip_relu
        self.skip_bn= cfg.score_skip_bn
        self.ratio_coef = cfg.score_multi_ratio
        self.budget_layers = cfg.budget_layers
        self.align_budget_layers = cfg.align_budget_layers
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger


    def ratio_score(self, stage_features_list, block_std_list):

        if len(stage_features_list)!=len(self.ratio_coef):
            raise ValueError("the length of the stage_features_list (%d) must be equal to the length of ratio_coef (%d)"%(
                            len(stage_features_list), self.ratio_coef))
        self.logger.debug("\nlen of stage_features_list:%d, len of block_std_list:%d\n%s"%(
                        len(stage_features_list), len(block_std_list), 
                        [np.log(std.item()) for std in block_std_list]))
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
                nas_score_std += torch.log(block_std_list[idx1])

            layer_num_idx = self.stage_layer_num[idx]
            
            # nas_score_feat = torch.sum(torch.abs(stage_features_list[idx]), dim=[1, 2, 3])
            # nas_score_feat = torch.log(torch.mean(nas_score_feat))
            # larger channel, larger performance.
            nas_score_feat = np.log(self.stage_channels[idx]) 
            # different stage with the different feature map ratio (2**(idx+6))/(4**(idx+1))
            nas_score_stage = nas_score_std + nas_score_feat
            # if self.align_budget_layers: nas_score_stage = nas_score_stage/layer_num_idx*self.budget_layers
            self.logger.debug("stage:%d, nas_score_stage:%.3f, score_feat:%.3f, log_std:%.3f"%(
                                idx, nas_score_stage.item(), nas_score_feat, nas_score_std.item()))

            nas_score_list.append(nas_score_stage.item()*ratio)
        self.logger.debug("nas_score:%s"%(np.sum(nas_score_list)))
        
        return nas_score_list


    def __call__(self, model):
        model.eval()
        model.requires_grad_(False)

        if self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            device = torch.device('cuda:{}'.format(self.gpu))
            model = model.cuda(self.gpu)
        else:
            device = torch.device('cpu')

        info = {}
        nas_score_list = []
        timer_start = time.time()
        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels = model.get_stage_info()

        for repeat_count in range(self.repeat):
            network_weight_gaussian_init(model, std=self.init_std)
            input = self.init_std*torch.randn(size=[self.batch_size, self.in_ch, self.resolution, self.resolution], device=device, dtype=torch.float32)
            # print("\ninitial input std: mean %.4f, std %.4f, max %.4f, min %.4f\n"%(
                    # input.mean().item(), input.std().item(), input.max().item(), input.min().item()))
            kwarg = {"init_std":self.init_std,}
            stage_features_list, block_std_list = model.entropy_forward_pre_GAP(input, skip_relu=self.skip_relu, skip_bn=self.skip_bn, **kwarg)
            nas_score_once = self.ratio_score(stage_features_list, block_std_list)
            nas_score_list.append(nas_score_once)

        timer_end = time.time()
        nas_score_list = np.array(nas_score_list)
        avg_nas_score = np.mean(np.sum(nas_score_list, axis=1))
        if self.align_budget_layers: 
            avg_nas_score = avg_nas_score/self.stage_layer_num[-1]*self.budget_layers
        std_nas_score = np.std(np.sum(nas_score_list, axis=1))

        info['avg_nas_score'] = avg_nas_score
        info['std_nas_score'] = std_nas_score
        info['nas_score_list'] = nas_score_list
        info['time'] = timer_end - timer_start
        self.logger.debug("avg_score:%s, consume time is %f ms\n"%(avg_nas_score, info['time']*1000))
        
        del model
        torch.cuda.empty_cache()    
        return info


def main():
    pass


if __name__ == '__main__':
    main()
    pass