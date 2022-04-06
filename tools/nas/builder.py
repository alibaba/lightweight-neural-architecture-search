# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

from logging import Logger
import os, sys
import warnings
import torch, thop
import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from models import __all_masternet__
from scores import __all_scores__
from latency import GetRobustLatencyMeanStd, OpProfiler
from configs import load_py_module_from_path 


class BuildNAS(metaclass=ABCMeta):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.build_master()
        self.build_space()
        self.build_score()
        self.build_latency()
        self.logger.info('****** Successfully build the NAS model ******\n')


    def build_master(self,):
        if hasattr(self.cfg, "space_arch"):
            self.AnyPlainNet = __all_masternet__[self.cfg.space_arch]
            self.logger.info("****** Build the masternet: %s ******"%(self.cfg.space_arch))
        else:
            raise NameError("cfg must have the parameter of 'space_arch'")


    def build_score(self,):
        if hasattr(self.cfg, "score_type"):
            self.compute_score = __all_scores__[self.cfg.score_type](self.cfg, logger=self.logger)
            self.logger.info("****** Build the score: %s ******"%(self.cfg.score_type))
        else:
            raise NameError("cfg must have the parameter of 'score_type'")


    def build_latency(self,):
        if self.cfg.lat_gpu:
            fp16 = True if self.cfg.lat_date_type=="FP16" else False
            self.benchmark_gpu = GetRobustLatencyMeanStd(self.cfg.lat_batch_size, self.cfg.resolution, 
                    self.cfg.gpu, channel=self.cfg.budget_image_channel, fp16=fp16)
            self.logger.info("****** Build the benchmark on searched GPU with %s ******"%(self.cfg.lat_date_type))

        if self.cfg.lat_pred:
            self.predictor = OpProfiler(device_name=self.cfg.lat_pred_device, date_type=self.cfg.lat_date_type, logger=self.logger)
            self.logger.info("****** Build the predictor on %s with %s ******"%(self.cfg.lat_pred_device, self.cfg.lat_date_type))
            

    def build_space(self,):
        if hasattr(self.cfg, "space_mutation"):
            mutation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spaces", "%s.py"%(self.cfg.space_mutation))
            assert os.path.isfile(mutation_path), "mutation_path is invalid"
            self.mutation = load_py_module_from_path("%s:mutate_function"%mutation_path)
            self.logger.info("****** Build the mutate_function: %s ******"%(self.cfg.space_mutation))
        else:
            raise NameError("cfg must have the parameter of 'space_mutation'")


    def do_compute_nas_score(self, model):

        try:
            nas_score_info = self.compute_score(model)
            the_nas_core = nas_score_info['avg_nas_score']

        except Exception as err:
            self.logger.error('!!! error in compute_score - rank%d!!!'%(self.cfg.rank))
            self.logger.error(str(err))
            self.logger.error('!!! Failed structure: ')
            self.logger.error(str(model.structure_info))
            the_nas_core = -9999   
                                                 
        return the_nas_core


    def do_benchmark(self, model):

        try:
            if self.cfg.lat_gpu and self.cfg.lat_pred:
                raise ValueError("lat_gpu and lat_pred in cfg cannot be equal to 1 at the same time")
            elif self.cfg.lat_gpu:
                the_latency = self.benchmark_gpu(model) # the unit is second
            elif self.cfg.lat_pred:
                net_params = model.get_params_for_trt(self.cfg.budget_image_size)

                # remove other params, only conv and convDW
                net_params_conv = []
                for idx, net_param in enumerate(net_params):
                    if net_param[0] in ["Regular", "Depthwise"]:
                        net_params_conv.append(net_param)
                times = [0]*len(net_params_conv)

                # the unit is millisecond with batch_size, so modify it to second
                _, the_latency = self.predictor(net_params_conv, times, self.cfg.lat_batch_size) 
                the_latency = the_latency/self.cfg.lat_batch_size/1000 
            else:
                the_latency = np.inf

        except Exception as e:
            self.logger.error('!!! error in benchmark_network - rank%d!!!'%(self.cfg.rank))
            self.logger.error(str(e))
            self.logger.error('!!! Failed structure: ')
            self.logger.error(str(model.structure_info))
            the_latency = np.inf

        return the_latency


    def is_satify_budget(self, model_info):
        if self.cfg.budget_layers is not None and self.cfg.budget_layers < model_info["layers"]:
            self.logger.debug('*** debug: rank={}, random structure too deep. \n  with the stucture={}'.format(self.cfg.rank, model_info))
            return False

        if self.cfg.budget_model_size is not None and self.cfg.budget_model_size < model_info["params"]:
            self.logger.debug('*** debug: rank={}, random structure params too large. \n  with the stucture={}'.format(self.cfg.rank, model_info))
            return False

        if self.cfg.budget_flops is not None and self.cfg.budget_flops < model_info["flops"]:
            self.logger.debug('*** debug: rank={}, random structure flops too large. \n  with the stucture={}'.format(self.cfg.rank, model_info))
            return False

        if self.cfg.budget_latency is not None and self.cfg.budget_latency < model_info["latency"]:
            self.logger.debug('*** debug: rank={}, random structure latency too large. \n  with the stucture={}'.format(self.cfg.rank, model_info))
            return False

        return True


    def get_info_for_evolution(self, structure_info=None, structure_str=None, structure_txt=None, flop_thop=False):

        model_info = {}

        model = self.AnyPlainNet(num_classes=self.cfg.space_num_classes, structure_info=structure_info, 
                structure_str=structure_str, structure_txt=structure_txt, block_module=self.cfg.space_block_module, 
                dropout_channel=self.cfg.space_dropout_channel, dropout_layer=self.cfg.space_dropout_layer, 
                out_indices=self.cfg.out_indices, classfication=self.cfg.space_classfication, 
                no_create=self.cfg.score_no_creat)

        if flop_thop:
            input_D = torch.randn(1, self.cfg.budget_image_channel, self.cfg.budget_image_size, self.cfg.budget_image_size)
            self.logger.info("the model is\n%s\n\n"%(model))
            flops_D, params_D = thop.profile(model, inputs=(input_D, ))
            flops_D, params_D = thop.clever_format([flops_D, params_D], "%.3f")
            self.logger.info('===> decoder:{}flops_{}params\n\n'.format(flops_D, params_D))
        
        model_info["structure_info"] = model.structure_info
        model_info["params"] = model.get_model_size()
        model_info["flops"] = model.get_flops(self.cfg.budget_image_size)
        model_info["layers"] = model.get_num_layers()
        model_info["stages"] = model.get_num_stages()
        model_info["latency"] = self.do_benchmark(model)
        model_info["is_satify_budget"] = self.is_satify_budget(model_info)

        if model_info["is_satify_budget"]: model_info["score"] = self.do_compute_nas_score(model)

        return model_info


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
