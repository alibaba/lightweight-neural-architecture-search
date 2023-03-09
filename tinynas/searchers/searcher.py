# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import argparse
import copy
import hashlib
import os
import pdb
import pickle
import random
import sys
import time
import warnings

import numpy as np
import torch

from filelock import FileLock
from tinynas.strategy import Strategy 
from tinynas.evolutions import Population
from tinynas.utils.file_utils import load_pyobj, save_pyobj 
from tinynas.utils.dist_utils import master_only, worker_only, get_dist_info,  is_master
from tinynas.utils.misc import clever_format 
from tinynas.utils.logger import get_root_logger
from .base import BaseSearcher
from .builder import SEARCHERS
from .synchonizer import Synchonizer

def check_duplicate(cfg, struct_info, logger=None):
    lock_file_name = os.path.join(cfg.work_dir, 'struct_set.lock')
    struct_set_file_name = os.path.join(cfg.work_dir, 'struct_set.txt')
    lock = FileLock(lock_file_name)
    logger.debug('waiting for the struct_set lock')
    with lock:
        logger.debug('get struct_set lock')
        # read struct set from lock file
        struct_set = set()
        if os.path.isfile(struct_set_file_name):
            with open(struct_set_file_name, 'r') as fin:
                lines = fin.readlines()
                struct_set = set([line.strip() for line in lines])
        logger.debug('read existed struct set, total num={}'.format(
            len(struct_set)))
        # format struct_info to a struct_str
        struct_str = str(struct_info).replace('\n', ' ').strip()
        # check if this struct_str is in struct_set
        if struct_str in struct_set:
            logger.debug(
                'input struct_info is duplicated, return directly')
            return True
        # if not, add this new struct_str into set, and write back to lock file
        struct_set.add(struct_str)
        logger.debug(
            'input struct_info is new, add it to struct_set and write back to file. total num={}'
            .format(len(struct_set)))
        with open(struct_set_file_name, 'w') as fout:
            fout.write('\n'.join(list(struct_set)))
        return False

@SEARCHERS.register_module(module_name='default_searcher')
class Searcher(BaseSearcher):
    def __init__(self, cfg_file, **kwargs):

        super().__init__(cfg_file)

        if 'cfg_options' in kwargs:
            self.cfg.merge_from_dict(kwargs['cfg_options'])

        self.log_freq = max(1,kwargs.get('log_freq', 500)) 
        self.cfg.rank, self.cfg.world_size = get_dist_info()
        self.cfg.gpu = None

        if hasattr(self.cfg, 'enable_gpu'):
            if self.cfg.enable_gpu and torch.cuda.is_available():
                self.cfg.gpu = self.cfg.rank %torch.cuda.device_count()

        random.seed(13+ self.cfg.rank)
        os.environ['PYTHONHASHSEED'] = str(13+ self.cfg.rank)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.cfg.work_dir, 'search_log/log_rank%d_%s' %
                                (self.cfg.rank, timestamp))
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = get_root_logger(
            name='TinyNAS',
            rank=self.cfg.rank,
            log_file=log_file,
            log_level=self.cfg.log_level)
        self.logger.info('Environment info:\n%s\n' % (str(self.cfg)))

        # copy config
        save_pyobj(os.path.join(self.cfg.work_dir, 'config_nas.txt'), self.cfg)

        # begin to build the super_model
        self.logger.info('begin to build the super model and population')
        self.strategy = Strategy(self.cfg, self.logger)

        self.popu_nas = Population(self.cfg.search.popu_size, self.strategy.budgets, self.logger)

        # load super_model and get the basic info
        super_model_info = self.strategy.get_info_for_evolution()
        self.init_structure_info = super_model_info['structure_info']
        self.logger.info(f'init: {super_model_info}')
        if not super_model_info['is_satify_budget']:
            raise ValueError(
                'The initial network must meet the limit budget, preferably less than 1/4'
            )

        # initialize the population with the super_model
        for i in range(self.popu_nas.popu_size):
            self.popu_nas.update_population(super_model_info)

        self.sync_interval = round(self.cfg.search.sync_size_ratio * self.cfg.search.popu_size)
        self.logger.info('sync_interval={}'.format(self.sync_interval))

        # rank 0 processes the merge task, so the iteration is smaller than others
        self.worker_max_iter = max(1, self.sync_interval // 10) if is_master() else self.sync_interval

        self.synchonizer = Synchonizer(
                                        self.cfg.world_size,
                                        self.cfg.search.num_random_nets,
                                        self.popu_nas, 
                                        self.sync_interval,
                                        self.worker_max_iter,
                                        self.logger)


    @master_only
    def export_cache_generation(self, last_export_generation_iteration):
        if self.popu_nas.num_evaluated_nets_count - last_export_generation_iteration >= self.log_freq:
            export_generation_filename = os.path.join(
                self.cfg.work_dir, 'nas_cache/iter{}.txt'.format(
                    self.popu_nas.num_evaluated_nets_count))
            self.logger.info('exporting generation: %s' %
                        (export_generation_filename))
            save_pyobj(export_generation_filename, self.popu_nas.export_dict())

            # logging intermediate results
            elasp_time = time.time() - self.start_time
            remain_time = elasp_time * float(
                self.cfg.search.num_random_nets - self.popu_nas.num_evaluated_nets_count) / (
                    1e-10 + float(self.popu_nas.num_evaluated_nets_count))
            if len(self.popu_nas.popu_acc_list) > 0:
                individual_info = self.popu_nas.get_individual_info(idx=0)
                # format
                for key, budget in self.strategy.budgets.items():
                    ratio = individual_info[key] / budget
                    value = clever_format(individual_info[key])
                    budget = clever_format(budget)
                    individual_info[key] = f'{ratio:.2%}, {value}/{budget}'
                self.logger.info(
                    'num_evaluated={}, elasp_time={:4g}h, remain_time={:4}h'.
                    format(self.popu_nas.num_evaluated_nets_count,
                           elasp_time / 3600, remain_time / 3600))
                self.logger.info('---best_individual: {}'.format(individual_info))

            last_export_generation_iteration = self.popu_nas.num_evaluated_nets_count
        # end export generation
        return last_export_generation_iteration

    @master_only 
    def export_result(self, ):
        # TODO: remove after debug
        export_dict = self.popu_nas.export_dict()
        val_dict = {
            'popu_structure_list': export_dict['popu_structure_list'],
            'popu_score_list': export_dict['popu_score_list']
        }
        dict_str = pickle.dumps(val_dict)
        hl = hashlib.md5()
        hl.update(dict_str)
        md5_str = hl.hexdigest()
        self.logger.info('search md5 str {}'.format(md5_str))

        # export final generation
        export_generation_filename = os.path.join(self.cfg.work_dir,
                                                  'nas_cache/iter_final.txt')
        self.logger.info('exporting generation: ' + export_generation_filename)
        save_pyobj(export_generation_filename, self.popu_nas.export_dict())

        # export best structure info
        if len(self.popu_nas.popu_acc_list) > 0:
            real_num_network = min(self.cfg.search.num_network,
                                   len(self.popu_nas.popu_acc_list))
            infos = []
            best_structures = []
            for i in range(real_num_network):
                individual_info = self.popu_nas.get_individual_info(
                    idx=i, is_struct=True)
                infos.append(individual_info)
                best_structures.append(individual_info['structure'])
            output_structures = {
                'space_arch': self.cfg.model.type,
                'best_structures': best_structures
            }
            best_structure_txt = os.path.join(self.cfg.work_dir, 'best_structure.txt')
            self.logger.info('exporting best_structure: ' + best_structure_txt)
            save_pyobj(best_structure_txt, output_structures)
            nas_info_txt = os.path.join(self.cfg.work_dir, 'nas_info.txt')
            self.logger.info('exporting nas_info: ' + nas_info_txt)
            save_pyobj(nas_info_txt, infos)

            # export subnet weights
            if hasattr(self.strategy.super_model, 'export'):
                ckpt_path = os.path.join(self.cfg.work_dir, 'weights/')
                os.makedirs(ckpt_path, exist_ok=True)
                self.logger.info('exporting best weights: ' + ckpt_path)
                for idx, structure in enumerate(best_structures):
                    state = self.strategy.super_model.export(structure)
                    torch.save(state, os.path.join(ckpt_path,
                                                   f'subnet{idx}.pt'))

        # remove struct duplicate files: lock and txt
        lock_file_name = os.path.join(self.cfg.work_dir, 'struct_set.lock')
        struct_set_file_name = os.path.join(self.cfg.work_dir, 'struct_set.txt')
        if os.path.isfile(lock_file_name):
            os.remove(lock_file_name)
        if os.path.isfile(struct_set_file_name):
            os.remove(struct_set_file_name)
        pass  # end with

    def search_step(self,
            popu_nas,
            strategy,
            logger=None,
            max_iter=None,
            cfg=None,
            init_structure_info=None,):
    
        # whether to fix the stage layer, enable minor_mutation for mutation function.
        if cfg.search.minor_mutation and popu_nas.num_evaluated_nets_count > cfg.search.minor_iter:
            minor_mutation = True
        else:
            minor_mutation = False
    
        for loop_count in range(max_iter):

            if len(popu_nas.popu_structure_list) > cfg.search.popu_size:
                logger.debug('*** debug: rank={}, population too large, remove some.'.format(cfg.rank))
                popu_nas.rank_population(maintain_popu=True)
            # ----- begin random generate a new structure and examine its performance ----- #
            logger.debug(
                'generate random structure, loop_count={}'.format(loop_count))
            if len(popu_nas.popu_structure_list) == 0:
                random_structure_info = init_structure_info
            else:
                init_random_structure_info = random.choice(
                    popu_nas.popu_structure_list)
                random_structure_info = strategy.mutation(
                    block_structure_info_list=init_random_structure_info,
                    minor_mutation=minor_mutation)
            pass  # end if
            logger.debug('random structure generated')
            #if check_duplicate(cfg, random_structure_info, logger):
            #    continue
    
            # load random_structure_info, get the basic info, update the population
            random_struct_info = strategy.get_info_for_evolution(
                structure_info=random_structure_info)
            if random_struct_info['is_satify_budget']:
                popu_nas.update_population(random_struct_info)
    
        pass  # end for loop_count
    
        popu_nas.rank_population(maintain_popu=True)
        logger.debug('return search_step ')
    
        self.popu_nas = popu_nas

    def search_loop(self, ): 
       
        self.start_time = time.time()
        last_export_generation_iteration = 0

        while True:

            enough_flag, self.popu_nas = self.synchonizer.sync_and_assign_jobs(self.popu_nas)
            if enough_flag:
                self.logger.debug('meet termination signal. Break now.')
                break

            self.logger.debug('search_step begin.')
            self.search_step(
                self.popu_nas,
                self.strategy,
                logger=self.logger,
                max_iter=self.worker_max_iter,
                cfg=self.cfg,
                init_structure_info=self.init_structure_info)

            self.logger.debug('search_step end.')

            # for worker node, push result to master
            self.popu_nas = self.synchonizer.sync_and_commit_result(self.popu_nas)
            last_export_generation_iteration = self.export_cache_generation(last_export_generation_iteration)

        pass  # end while True

        # export results for master node
    def run(self, *args, **kwargs):

        self.search_loop()
        self.export_result()
        #exit()

