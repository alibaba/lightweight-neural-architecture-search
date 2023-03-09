# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import argparse
import ast
import os
from collections.abc import Iterable
import numpy as np

def clever_format(nums, format='%.2f'):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + 'T')
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + 'G')
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + 'M')
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + 'K')
        else:
            clever_nums.append(num)
            # clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


def filter_dict_list(dict_list, **kwargs):
    new_list = dict_list

    for key, value in kwargs.items():
        if len(new_list) == 0:
            return []
        new_list = [
            x for x in new_list
            if (isinstance(x[key], float) and abs(x[key] - value) < 1e-6)
            or x[key] == value
        ]

    return new_list

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

class LearningRateScheduler():

    def __init__(
        self,
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
            return float(num_received_training_instances
                         + 1) / float(warmup_instances) * self.lr

        ratio_epoch = float(num_received_training_instances - warmup_instances
                            + 1) / float(stop_instances - warmup_instances)

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
