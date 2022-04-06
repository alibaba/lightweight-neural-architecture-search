# Copyright (c) 2021-2022 Alibaba Group Holding Limited.
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Origin file: 
# https://github.com/Megvii-BaseDetection/YOLOX/blob/b861b22d3f8c78bfab129b8bbfbe2875822941e0/yolox/exp/base_exp.py

import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate


class BaseConfig(metaclass=ABCMeta): # modify in 2021 by Zhenhong Sun
    """Basic class for any config."""

    def __init__(self):
        self.seed = None
        self.work_dir = "./outputs"  # modify in 2021 by Zhenhong Sun

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        for k, v in cfg_list.items(): # modify in 2021 by Zhenhong Sun
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)


if __name__ == '__main__':
    pass