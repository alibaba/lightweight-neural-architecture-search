# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import numpy as np
from abc import ABCMeta, abstractmethod


class Population(metaclass=ABCMeta):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger=logger
        self.popu_size = cfg.ea_popu_size
        self.init_population()
        self.logger.info('****** Successfully build the Population ******')


    def init_population(self, ):
        self.num_evaluated_nets_count = 0
        self.popu_structure_list = []
        self.popu_acc_list = []
        self.popu_score_list = []
        self.popu_params_list = []
        self.popu_flops_list = []
        self.popu_latency_list = []
        self.popu_layers_list = []
        self.popu_stages_list = []


    def update_population(self, model_info):
        if "score" not in model_info.keys():
            raise NameError("To update population, score must in the model_info")
            
        if self.cfg.score_flop_ratio is not None:
            acc_temp = model_info["score"] + self.cfg.score_flop_ratio*model_info["flops"]
        else:
            acc_temp = model_info["score"]

        if len(self.popu_acc_list)>0:
            insert_idx = len(self.popu_acc_list) if self.popu_acc_list[-1]<acc_temp else 0
            for idx, pupu_acc in enumerate(self.popu_acc_list):
                if pupu_acc>=acc_temp:
                    insert_idx = idx
        else:
            insert_idx = 0

        self.popu_structure_list.insert(insert_idx, model_info["structure_info"])
        self.popu_acc_list.insert(insert_idx, acc_temp)
        self.popu_score_list.insert(insert_idx, model_info["score"])
        self.popu_params_list.insert(insert_idx, model_info["params"])
        self.popu_flops_list.insert(insert_idx, model_info["flops"])
        self.popu_latency_list.insert(insert_idx,  model_info["latency"])
        self.popu_layers_list.insert(insert_idx,  model_info["layers"])
        self.popu_stages_list.insert(insert_idx,  model_info["stages"])


    def rank_population(self, maintain_popu=False):
        # filter out the duplicate structure
        unique_structure_set = set()
        unique_idx_list = []
        for the_idx, the_strucure in enumerate(self.popu_structure_list):
            if str(the_strucure) in unique_structure_set:
                continue
            unique_structure_set.add(str(the_strucure))
            unique_idx_list.append(the_idx)

        # sort population list, pop the duplicate structure, and maintain the population
        sort_idx = list(np.argsort(self.popu_acc_list))
        sort_idx = sort_idx[::-1]
        for idx in sort_idx:
            if idx not in unique_idx_list:
                sort_idx.remove(idx)
        if maintain_popu: sort_idx = sort_idx[0:self.popu_size]
        
        self.popu_structure_list = [self.popu_structure_list[idx] for idx in sort_idx]
        self.popu_acc_list = [self.popu_acc_list[idx] for idx in sort_idx]
        self.popu_score_list = [self.popu_score_list[idx] for idx in sort_idx]
        self.popu_params_list = [self.popu_params_list[idx] for idx in sort_idx]
        self.popu_flops_list = [self.popu_flops_list[idx] for idx in sort_idx]
        self.popu_latency_list = [self.popu_latency_list[idx] for idx in sort_idx]
        self.popu_layers_list = [self.popu_layers_list[idx] for idx in sort_idx]
        self.popu_stages_list = [self.popu_stages_list[idx] for idx in sort_idx]


    def gen_random_structure_net(self,):
        pass


    def merge_shared_data(self, popu_nas_info, update_num=True):

        if isinstance(popu_nas_info, Population):
            self.popu_structure_list += popu_nas_info.popu_structure_list
            self.popu_acc_list += popu_nas_info.popu_acc_list
            self.popu_score_list += popu_nas_info.popu_score_list
            self.popu_params_list += popu_nas_info.popu_params_list
            self.popu_flops_list += popu_nas_info.popu_flops_list
            self.popu_latency_list += popu_nas_info.popu_latency_list
            self.popu_layers_list += popu_nas_info.popu_layers_list
            self.popu_stages_list += popu_nas_info.popu_stages_list


        if isinstance(popu_nas_info, dict):
            if update_num: self.num_evaluated_nets_count = popu_nas_info["num_evaluated_nets_count"]
            self.popu_structure_list += popu_nas_info["popu_structure_list"]
            self.popu_acc_list += popu_nas_info["popu_acc_list"]
            self.popu_score_list += popu_nas_info["popu_score_list"]
            self.popu_params_list += popu_nas_info["popu_params_list"]
            self.popu_flops_list += popu_nas_info["popu_flops_list"]
            self.popu_latency_list += popu_nas_info["popu_latency_list"]
            self.popu_layers_list += popu_nas_info["popu_layers_list"]
            self.popu_stages_list += popu_nas_info["popu_stages_list"]

        self.rank_population(maintain_popu=True)


    def export_dict(self,):
        popu_nas_info = {}
        self.rank_population(maintain_popu=True)

        popu_nas_info["num_evaluated_nets_count"] = self.num_evaluated_nets_count
        popu_nas_info["popu_structure_list"] = self.popu_structure_list
        popu_nas_info["popu_acc_list"] = self.popu_acc_list
        popu_nas_info["popu_score_list"] = self.popu_score_list
        popu_nas_info["popu_params_list"] = self.popu_params_list
        popu_nas_info["popu_flops_list"] = self.popu_flops_list
        popu_nas_info["popu_latency_list"] = self.popu_latency_list
        popu_nas_info["popu_layers_list"] = self.popu_layers_list
        popu_nas_info["popu_stages_list"] = self.popu_stages_list
        
        return popu_nas_info


    def get_individual_info(self, idx=0, is_struct=False):
        individual_info = {}
        self.rank_population(maintain_popu=True)
        
        if is_struct: individual_info["structure"] = self.popu_structure_list[idx]
        individual_info["acc"] = self.popu_acc_list[idx]
        individual_info["score"] = self.popu_score_list[idx]
        individual_info["params"] = self.popu_params_list[idx]
        individual_info["flops"] = self.popu_flops_list[idx]
        individual_info["latency"] = self.popu_latency_list[idx]
        individual_info["layers"] = self.popu_layers_list[idx]
        individual_info["stages"] = self.popu_stages_list[idx]
        
        return individual_info