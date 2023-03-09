# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np


class Population(metaclass=ABCMeta):

    def __init__(self, popu_size, budgets, logger):
        self.logger = logger
        self.budgets = budgets
        self.popu_size = popu_size
        self.init_population()
        self.logger.info('****** Successfully build the Population ******')

    def init_population(self, ):
        self.num_evaluated_nets_count = 0
        self.popu_structure_list = []
        self.popu_acc_list = []
        self.popu_score_list = []
        for key in self.budgets:
            temp_popu_list = list()
            setattr(self, f'popu_{key}_list', temp_popu_list)

    def update_population(self, model_info):
        if 'score' not in model_info.keys():
            raise NameError(
                'To update population, score must in the model_info')

        acc_temp = model_info['score']

        if len(self.popu_acc_list) > 0:
            insert_idx = len(
                self.popu_acc_list) if self.popu_acc_list[-1] < acc_temp else 0
            for idx, pupu_acc in enumerate(self.popu_acc_list):
                if pupu_acc >= acc_temp:
                    insert_idx = idx
        else:
            insert_idx = 0

        self.popu_structure_list.insert(insert_idx,
                                        model_info['structure_info'])
        self.popu_acc_list.insert(insert_idx, acc_temp)
        self.popu_score_list.insert(insert_idx, model_info['score'])
        for key in self.budgets:
            _popu_list = getattr(self, f'popu_{key}_list')
            _popu_list.insert(insert_idx, model_info[key])


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
        temp_sort_idx = sort_idx[:]
        for idx in temp_sort_idx:
            if idx not in unique_idx_list:
                sort_idx.remove(idx)
        if maintain_popu:
            sort_idx = sort_idx[0:self.popu_size]

        self.popu_structure_list = [
            self.popu_structure_list[idx] for idx in sort_idx
        ]
        self.popu_acc_list = [self.popu_acc_list[idx] for idx in sort_idx]
        self.popu_score_list = [self.popu_score_list[idx] for idx in sort_idx]

        for key in self.budgets:
            popu_list_name = f'popu_{key}_list'
            temp_popu_list = getattr(self, popu_list_name)
            setattr(self, popu_list_name,
                    [temp_popu_list[idx] for idx in sort_idx])

    def gen_random_structure_net(self, ):
        pass

    def merge_shared_data(self, popu_nas_info, update_num=True):

        if isinstance(popu_nas_info, Population):
            self.popu_structure_list += popu_nas_info.popu_structure_list
            self.popu_acc_list += popu_nas_info.popu_acc_list
            self.popu_score_list += popu_nas_info.popu_score_list
            for key in self.budgets:
                popu_list_name = f'popu_{key}_list'
                righthand = getattr(popu_nas_info, popu_list_name)
                lefthand = getattr(self, popu_list_name)
                lefthand += righthand
                setattr(self, popu_list_name, lefthand)

        if isinstance(popu_nas_info, dict):
            if update_num:
                self.num_evaluated_nets_count = popu_nas_info[
                    'num_evaluated_nets_count']
            self.popu_structure_list += popu_nas_info['popu_structure_list']
            self.popu_acc_list += popu_nas_info['popu_acc_list']
            self.popu_score_list += popu_nas_info['popu_score_list']
            for key in self.budgets:
                popu_list_name = f'popu_{key}_list'
                righthand = popu_nas_info[f'popu_{key}_list']
                lefthand = getattr(self, popu_list_name)
                lefthand += righthand
                setattr(self, popu_list_name, lefthand)

        self.rank_population(maintain_popu=True)

    def export_dict(self, ):
        popu_nas_info = {}
        self.rank_population(maintain_popu=True)

        popu_nas_info[
            'num_evaluated_nets_count'] = self.num_evaluated_nets_count
        popu_nas_info['popu_structure_list'] = self.popu_structure_list
        popu_nas_info['popu_acc_list'] = self.popu_acc_list
        popu_nas_info['popu_score_list'] = self.popu_score_list
        for key in self.budgets:
            popu_nas_info[f'popu_{key}_list'] = getattr(
                self, f'popu_{key}_list')

        return popu_nas_info

    def get_individual_info(self, idx=0, is_struct=False):
        individual_info = {}
        self.rank_population(maintain_popu=True)

        if is_struct:
            individual_info['structure'] = self.popu_structure_list[idx]
        individual_info['acc'] = self.popu_acc_list[idx]
        individual_info['score'] = self.popu_score_list[idx]
        for key in self.budgets:
            individual_info[key] = getattr(self, f'popu_{key}_list')[idx]

        return individual_info
