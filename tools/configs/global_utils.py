# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import numpy as np
import torch
import pprint, ast, argparse, logging
import distutils.dir_util

from utils import get_logger, acquire_gpu, release_gpu


def filter_dict_list(dict_list, **kwargs):
    new_list = dict_list

    for key, value in kwargs.items():
        if len(new_list) == 0:
            return []
        new_list = [x for x in new_list if (isinstance(x[key], float) and abs(x[key] - value) < 1e-6) or  x[key] == value]

    return new_list

def load_py_module_from_path(module_path, module_name=None):
    if module_path.find(':') > 0:
        split_path = module_path.split(':')
        module_path = split_path[0]
        function_name = split_path[1]
    else:
        function_name = None

    if module_name is None:
        module_name = module_path.replace('/', '_').replace('.', '_')

    assert os.path.isfile(module_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    any_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(any_module)
    if function_name is None:
        return any_module
    else:
        return getattr(any_module, function_name)


def smart_round(x, base=None):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)


def auto_assign_gpu():
    # auto assign gpu

    num_total_gpus = torch.cuda.device_count()
    gpu_id_list = [str(x) for x in list(range(num_total_gpus))]
    new_acquire_gpu_id_list = acquire_gpu(gpu_id_list=gpu_id_list, num_acq_gpus=1)
    gpu = int(new_acquire_gpu_id_list[0])
    torch.cuda.set_device(gpu)
    return gpu


def release_gpu_all(release_gpu_id_list):
    if isinstance(release_gpu_id_list, int):
        release_gpu_id_list = [str(release_gpu_id_list)]
    elif isinstance(release_gpu_id_list, str):
        release_gpu_id_list = [release_gpu_id_list]
    elif isinstance(release_gpu_id_list, list):
        release_gpu_id_list = [str(x) for x in release_gpu_id_list]
    else:
        raise ValueError('release_gpu_id_list must be int, str, or list of str/int')

    release_gpu(release_gpu_id_list)


class AutoGPU:
    def __init__(self):
        self.gpu = auto_assign_gpu()
    def __del__(self):
        release_gpu_all(self.gpu)
        


def merge_object_attr(obj1, obj2):
    new_dict = {}
    for k, v in obj1.__dict__.items():
        if v is None and k in obj2.__dict__:
            new_v = obj2.__dict__[k]
        else:
            new_v = v
        new_dict[k] = new_v

    for k, v in obj2.__dict__.items():
        if k not in new_dict:
            new_dict[k] = v

    obj1.__dict__.update(new_dict)
    return obj1


def smart_float(str1):
    if str1 is None:
        return None
    the_base = 1
    if str1[-1] == 'k':
        the_base = 1000
        str1 = str1[0:-1]
    elif str1[-1] == 'm':
        the_base = 1000000
        str1 = str1[0:-1]
    elif str1[-1] == 'g':
        the_base = 1000000000
        str1 = str1[0:-1]
    pass
    the_x = float(str1) * the_base
    return the_x

def split_str_to_list(str_to_split):
    group_str = str_to_split.split(',')
    the_list = []
    for s in group_str:
        t = s.split('*')
        if len(t) == 1:
            the_list.append(s)
        else:
            the_list += [t[0]] * int(t[1])
    return the_list

def mkfilepath(filename):
    filename = os.path.expanduser(filename)
    distutils.dir_util.mkpath(os.path.dirname(filename))


def mkdir(dirname):
    dirname = os.path.expanduser(dirname)
    distutils.dir_util.mkpath(dirname)


def robust_save(filename, save_function):
    mkfilepath(filename)
    backup_filename = filename + '.robust_save_temp'
    save_function(backup_filename)
    if os.path.isfile(filename):
        os.remove(filename)
    os.rename(backup_filename, filename)


def save_pyobj(filename, pyobj):
    mkfilepath(filename)
    the_s = pprint.pformat(pyobj, indent=2, width=120, compact=True)
    with open(filename, 'w') as fid:
        fid.write(the_s)


def load_pyobj(filename):
    with open(filename, 'r') as fid:
        the_s = fid.readlines()

    if isinstance(the_s, list):
        the_s = ''.join(the_s)

    the_s = the_s.replace('inf', '1e20')
    pyobj = ast.literal_eval(the_s)
    return pyobj


def get_root_logger(name='Search', rank=0, log_file=None, log_level=logging.INFO):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to 'nas'.
        log_file ([type], optional): [description]. Defaults to None.
        log_level ([type], optional): [description]. Defaults to logging.INFO.

    Returns:
        [type]: [description]
    """
    logger = get_logger(name=name, rank=rank, log_file=log_file, log_level=log_level)

    return logger


class MyLogger():
    def __init__(self, log_filename=None, verbose=False):
        self.log_filename = log_filename
        self.verbose = verbose
        if self.log_filename is not None:
            mkfilepath(self.log_filename)
            self.fid = open(self.log_filename, 'w')
        else:
            self.fid = None

    def info(self, msg):
        msg = str(msg)
        print(msg)
        if self.fid is not None:
            self.fid.write(msg + '\n')
            self.fid.flush()

    def debug_info(self, msg):
        if not self.verbose:
            return
        msg = str(msg)
        print(msg)
        if self.fid is not None:
            self.fid.write(msg + '\n')
            self.fid.flush()

class LearningRateScheduler():
    def __init__(self,
                 mode,
                 lr,
                 target_lr=None,
                 num_training_instances=None,
                 stop_epoch=None,
                 warmup_epoch=None,
                 stage_list=None,
                 stage_decay=None,
                 ):
        self.mode = mode
        self.lr = lr
        self.target_lr = target_lr if target_lr is not None else 0
        self.num_training_instances = num_training_instances if num_training_instances is not None else 1
        self.stop_epoch = stop_epoch if stop_epoch is not None else np.inf
        self.warmup_epoch = warmup_epoch if warmup_epoch is not None else 0
        self.stage_list = stage_list if stage_list is not None else None
        self.stage_decay = stage_decay if stage_decay is not None else 0

        self.num_received_training_instances = 0

        if self.stage_list is not None:
            self.stage_list = [int(x) for x in self.stage_list.split(',')]

    def update_lr(self, batch_size):
        self.num_received_training_instances += batch_size

    def get_lr(self, num_received_training_instances=None):
        if num_received_training_instances is None:
            num_received_training_instances = self.num_received_training_instances

        # start_instances = self.num_training_instances * self.start_epoch
        stop_instances = self.num_training_instances * self.stop_epoch
        warmup_instances = self.num_training_instances * self.warmup_epoch

        assert stop_instances > warmup_instances

        current_epoch = self.num_received_training_instances // self.num_training_instances

        if num_received_training_instances < warmup_instances:
            return float(num_received_training_instances + 1) / float(warmup_instances) * self.lr

        ratio_epoch = float(num_received_training_instances - warmup_instances + 1) / \
                      float(stop_instances - warmup_instances)

        if self.mode == 'cosine':
            factor = (1 - np.math.cos(np.math.pi * ratio_epoch)) / 2.0
            return self.lr + (self.target_lr - self.lr) * factor
        elif self.mode == 'stagedecay':
            stage_lr = self.lr
            for stage_epoch in self.stage_list:
                if current_epoch <= stage_epoch:
                    return stage_lr
                else:
                    stage_lr *= self.stage_decay
                pass  # end if
            pass  # end for
            return stage_lr
        elif self.mode == 'linear':
            factor = ratio_epoch
            return self.lr + (self.target_lr - self.lr) * factor
        else:
            raise RuntimeError('Unknown learning rate mode: ' + self.mode)
        pass  # end if