# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import pdb
import time
import copy
import random
import warnings
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import (get_root_logger, load_py_module_from_path, 
                AutoGPU, load_pyobj, save_pyobj, DictAction)
from nas.builder import BuildNAS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def get_single_info(cfg):
    # args = parse_args()
    cfg.rank = 0
    cfg.gpu = 0
    log_file = os.path.join(cfg.work_dir, "master.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = get_root_logger(name='Masternet Info', rank=0, log_file=log_file, log_level=cfg.log_level)
    logger.info('Environment info:\n%s\n'%(str(cfg)))

    # begin to build the masternet
    logger.info('begin to build the masternet and population:\n')
    model_nas = BuildNAS(cfg, logger)

    # load masternet and get the basic info
    masternet_info = model_nas.get_info_for_evolution(structure_txt=cfg.space_structure_txt, flop_thop=False)
    logger.info(masternet_info)

    return masternet_info

if __name__ == '__main__':
    masternet_info = get_single_info()