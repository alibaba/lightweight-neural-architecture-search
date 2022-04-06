# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
import random
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_base import BaseConfig

class Config(BaseConfig):
    def __init__(self):
        super().__init__()

        """ Task config """
        self.task = "detection" # classification, detection
        self.space_classfication = False # True for single out, False for out_indices
        self.out_indices=(1, 2, 3, 4) # output stages starts from 0 
        self.work_dir = "./example" # work_dir to save the results
        self.log_level = "INFO" # INFO/DEBUG
        self.only_master = False # no search, only show the masternet info

        """ Budget config, None is not constrained """
        # the minimum value is 128, the maximum value is 480 for latency prediction.
        self.budget_image_size = 224 
        self.budget_image_channel = 3
        self.budget_model_size = None # the number of parameters
        self.budget_flops = None # the FLOPs similar to thop
        self.budget_latency = None # the unit is second
        self.budget_layers = 49 # 49 for R50, 100 for R101
        self.budget_stages = 5 # Downsample

        """ Score config """
        self.score_type = "entropy" # entropy 
        self.score_batch_size = 32 #
        # 224 for Imagenet, 480 for detection, 160 for mcu
        self.score_image_size = 224 
        self.score_image_channel = 3
        self.align_budget_layers = False # reserved
        # score params for entropy score
        self.score_no_creat = False # False
        self.score_repeat = 4 #
        self.score_skip_relu = True # no relu in forward
        self.score_skip_bn = True # no bn in forward
        self.score_multi_ratio = [0, 0, 1, 1, 6] # weight ratio of 5 downsampling stages
        self.score_init_std = 1 # initialization std
        # Score params for adding other constraits to the ACC
        self.score_flop_ratio = None # Acc = score + ration*flops/1e6

        """ Latency config """
        self.lat_gpu = False # whether to mearsure the latency with gpu
        self.lat_pred = False # whether to predictor the latency
        self.lat_date_type = "FP16" # FP32, FP16, INT8
        self.lat_pred_device = "V100" # V100, t40
        self.lat_batch_size = 32 # latency batch size
        self.lat_repeat = 1 # reserved

        """ Search Space config """
        self.space_arch = "MasterNet"
        self.space_mutation = "space_K1KXK1"
        self.space_block_num = 2 # mutate x blocks once
        self.space_num_classes = 1000 # classfication number, not need for others
        self.space_structure_txt = None # init_structure 
        self.space_exclude_stem = False # exclude the stem block for mutation 
        self.space_block_module = None # extra block not defined in models
        self.space_minor_mutation = False # whether fix the stage layer
        self.space_minor_iter = 100000 # which iteration to enable minor_mutation
        self.space_dropout_channel = None # reserved
        self.space_dropout_layer = None # reserved
        self.space_structure_str = None # reserved

        """ EA config """
        self.ea_dist_mode = "mpi" # single or mpi
        self.ea_popu_size = 256 # the populaiton size
        self.ea_log_freq = 1000 # the interval for show results
        self.ea_num_random_nets = 100000 # the searching iterations
        self.ea_sync_size_ratio = 1.0 # control each thread sync number: ratio * popu_size
        self.ea_load_population = None # whether load searched population

        """ check the valid of config """
        # self.config_check()


    def config_check(self):
        valid_tasks = ["classification", "detection", "mcu", "action"]
        if hasattr(self, "task"):
            if self.task not in valid_tasks:
                raise ValueError("Task name must be in %s"%(valid_tasks))

        if self.log_level.upper() == "DEBUG": 
            self.log_level = logging.DEBUG
        elif self.log_level.upper() == "INFO": 
            self.log_level = logging.INFO
        else:
            raise ValueError("Supported log_level is INFO of DEBUG, not %s"%(self.log_level))

        if self.budget_image_size < 128:
            raise ValueError("Budget_image_size must be larger than 128, not %d"%(self.budget_image_size))
        if self.lat_pred is True:
            if self.lat_gpu is True:
                raise ValueError("Latency must be benchmarkd on gpu or prediction, please check that")
            if self.budget_image_size > 480:
                raise ValueError("Budget_image_size must be less than 480 when using latency prediction, not %d"%(self.budget_image_size))
        
        if len(self.score_multi_ratio)!=self.budget_stages:
            raise ValueError("The length of score_multi_ratio must be equal to budget_stages, please check that")
        if self.budget_model_size=="None": self.budget_model_size = None # the number of parameters
        if self.budget_flops=="None": self.budget_flops = None # the FLOPs similar to thop
        if self.budget_latency=="None": self.budget_latency = None # the unit is second


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    pass