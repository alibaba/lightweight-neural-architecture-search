# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import ast
import logging
import os
import sys

from torch import nn
from abc import ABC, abstractmethod

class Model(ABC, nn.Module):

    def __init__(self,
                 structure=None,
                 logger=None):
        super().__init__()
        self.structure_info = structure
        self.logger = logger or logging

        if isinstance(structure, str):
            if os.path.exists(structure):
                with open(structure, 'r') as fid:
                    self.structure_info = ''.join(
                        [x.strip() for x in fid.readlines()])
            self.structure_info = ast.literal_eval(self.structure_info)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass 

    @abstractmethod
    def get_model_size(self, return_list=False):
        pass 

    @abstractmethod
    def get_flops(self, resolution):
        pass

    @abstractmethod
    def build(self, structure_info):
        pass 
