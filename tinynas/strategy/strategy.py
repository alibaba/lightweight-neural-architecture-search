# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import os
from abc import ABCMeta

import numpy as np
import thop
import torch
from tinynas.spaces import build_space
from tinynas.scores import build_score
from tinynas.models import build_model
from tinynas.budgets import build_budget
from tinynas.latency import build_latency

class Strategy(metaclass=ABCMeta):

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.build_budgets()
        self.build_space()
        self.build_model()
        self.build_score()
        self.build_latency()
        self.logger.info('****** Successfully build the NAS Strategy ******')

    def build_space(self, ):
        if hasattr(self.cfg, 'space'):
            space_cfg = self.cfg.space
            if 'layers' in self._budget_values:
                space_cfg['budget_layers'] = self._budget_values['layers']
            space_cfg['name'] = space_cfg.type
            self.mutation = build_space(space_cfg)
            self.choices = self.mutation.choice() 
            self.logger.info('****** Build the mutation space: %s ******' %
                             (space_cfg.type))
        else:
            raise NameError("cfg must have the parameter of 'space'")

    def build_model(self, ):
        if hasattr(self.cfg, 'model'):
            model_cfg = self.cfg.model
            model_cfg['choices'] = self.choices
            model_cfg['logger'] = self.logger
            self.super_model = build_model(model_cfg)
            self.logger.info('****** Build the super_model:%s ******'%(model_cfg.type))
        else:
            raise NameError("cfg must have the parameter of 'model'")

    def build_score(self, ):
        if hasattr(self.cfg, 'score'):
            score_cfg = self.cfg.score
            default_args = dict( 
                    cfg =  self.cfg, 
                    logger  = self.logger
                    )
            self.compute_score= build_score(score_cfg, default_args = default_args)
            self.logger.info('****** Build the score: %s ******' %
                             (score_cfg.type))
        else:
            raise NameError("cfg must have the parameter of 'score_type'")

    def build_budgets(self,):
        self._budgets = []
        budget_cfg = self.cfg.budgets
        assert isinstance(budget_cfg, list)
        for cfg_i in budget_cfg:
            cfg_i['logger'] = self.logger
            self._budgets.append(build_budget(cfg_i))
        self._budget_values = dict()
        for item in self._budgets:
            self._budget_values[item.name] = item.budget

    def build_latency(self, ):
        self.latency_func = None
        if hasattr(self.cfg, 'latency'):
            latency_cfg = self.cfg.latency
            default_args = dict( 
                    gpu = self.cfg.gpu,
                    logger  = self.logger
                    )
            self.latency_func = build_latency(latency_cfg, default_args = default_args)
            self.logger.info('****** Build the latency: %s ******' %
                             (latency_cfg.type))
        else:
            assert 'latency' not in self.budgets, 'cfg must have the parameter of latency cfg when used latency in budget'

    @property
    def budgets(self,):
        return self._budget_values
        
    def do_compute_nas_score(self, model):
        try:
            nas_score_info = self.compute_score(model)
            the_nas_score = nas_score_info['avg_nas_score']

        except Exception as err:
            self.logger.error('error in compute_score')
            # self.logger.error(str(err))
            self.logger.exception(err)
            self.logger.error('Failed structure: ')
            self.logger.error(str(model.structure_info))
            the_nas_score = -9999

        return the_nas_score

    def is_satify_budget(self, model_info):
        for budget_i in self._budgets:
            if not budget_i(model_info):
                return False
        return True

    def get_info_for_evolution(self, structure_info=None):
        model = self.super_model.build(structure_info)

        model_info = {'structure_info': model.structure_info}

        for key in self._budget_values:
            kwargs = {}
            if key in {'flops', 'max_feature'}:
                kwargs['resolution'] = self.cfg.image_size
                if 'frames' in self.cfg:
                    kwargs['frames'] = self.cfg.frames
            if key in {'latency'}:
                kwargs['latency_func'] = self.latency_func
            model_info[key] = getattr(model, 'get_' + key)(**kwargs)
        
        model_info['is_satify_budget'] = self.is_satify_budget(model_info)

        if model_info['is_satify_budget']:
            model_info['score'] = self.do_compute_nas_score(model)
            depth_penalty_ratio = self.cfg.score.get('depth_penalty_ratio', 0) 
            if self.cfg.score.type == 'deepmad' and depth_penalty_ratio != 0:
                depth_list_every_block = []
                for block in model_info["structure_info"]:
                    if 'L' in block:
                        depth_list_every_block.append(int(block['L']))
                if model_info['structure_info'][1]['class'] == 'SuperResK1KXK1':
                    depth_list_every_block = depth_list_every_block[1:]
                    assert len(depth_list_every_block) == 4
                depth_list_every_block = np.array(depth_list_every_block)

                # remove the first block because the first block needs <= 3
                if model_info["structure_info"][1]['class'] in ["SuperResK1KXK1", "SuperResKXKX"]:
                    depth_list_every_block = depth_list_every_block[1:]

                depth_uneven_score = np.exp(np.std(depth_list_every_block))
                depth_penalty = depth_uneven_score * depth_penalty_ratio
                model_info['score'] = model_info['score'] - depth_penalty

        return model_info
