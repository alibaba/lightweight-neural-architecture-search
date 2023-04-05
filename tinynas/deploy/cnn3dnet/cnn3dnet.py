'''
example:
--arch AdamNAS/masternet.py:MasterNet \
--block_module AdamNAS/blocks_v3.py \
--structure_txt ${save_dir}/init_structure.txt \
--num_classes 100 \
--NAS_mode \

block structure example: ['conv3x3', ${in_channels}, ${out_channels}, ${stride}]
'''
import argparse
import ast
import logging
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .blocks_cnn_3d import __all_blocks_3D__, network_weight_stupid_init, Swish
from .builder import MODELS 
from .base import Model




def __interpolate_channel__(input, out_channels):
    b, c, h, w = input.shape

    if out_channels > c:
        n = out_channels // c
        input_list = [input] * n
        output = torch.cat(input_list, dim=1)
    else:
        n = c // out_channels
        output = input.view(b, n, out_channels, h, w)
        output = torch.mean(output, dim=1)

    return output


def load_model(model, 
               load_parameters_from, 
               strict_load=False, 
               map_location=torch.device('cpu'), 
               **kwargs):
    if not os.path.isfile(load_parameters_from):
        raise ValueError("bad checkpoint to load %s"%(load_parameters_from))
    else:
        print('loading params from ' + load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict_load)

    return model

class Cnn3DNet(Model):

    def __init__(self, 
                 num_classes=None,
                 num_hidden = None,
                 structure_info=None,
                 dropout_channel=None, 
                 dropout_layer=None, 
                 out_indices=(1, 2, 3, 4),
                 no_create=False,
                 classfication=False,
                 pretrained=None,
                 logger=None,
                 **kwargs):
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.structure_info = structure_info
        # self.structure_str = structure_str
        # self.structure_txt = structure_txt
        # self.block_module = block_module
        self.out_indices = out_indices
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer
        self.classfication = classfication
        self.__all_blocks__ = __all_blocks_3D__
        assert structure_info, 'Must set structure_info'

        super().__init__(structure_info, logger)
        assert isinstance(self.structure_info, list)
        self.no_create = no_create

        self.block_list = nn.ModuleList()
        for block_structure_info in self.structure_info:
            the_block_class = self.__all_blocks__[
                block_structure_info['class']]
            the_block = the_block_class(
                block_structure_info, no_create=self.no_create, **kwargs)
            self.block_list.append(the_block)
        pass

        if self.classfication: # output for the action classfication task
            self.fc_linear = nn.Sequential(
                            nn.Linear(self.block_list[-1].out_channels, self.num_hidden, bias=True), 
                            Swish(),
                            nn.Dropout(p=0.2, inplace=True),
                            nn.Linear(self.num_hidden, self.num_classes, bias=True))

        self.block_list = nn.ModuleList()
        for block_structure_info in self.structure_info:
            the_block_class = self.__all_blocks__[block_structure_info['class']]
            the_block = the_block_class(block_structure_info, no_create=self.no_create)
            self.block_list.append(the_block)
        pass

        

        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels, _ = self.get_stage_info()
        # set dropout rate
        L = self.get_layers()
        current_depth = 0
        for block in self.block_list:
            current_depth += block.get_layers()
            if self.dropout_channel is not None:
                block.dropout_channel = self.dropout_channel * current_depth / L
            if self.dropout_layer is not None:
                block.dropout_layer = self.dropout_layer * current_depth / L
        
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of masternet.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_model(self, pretrained, strict=False)
 
        elif pretrained is None:
            pass


    def forward(self, x):
        # add different stages outputs for detection
        output = x
        stage_idx_output = [self.stage_idx[idx] for idx in self.out_indices]
        stage_features_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in stage_idx_output:
                stage_features_list.append(output)

        if self.classfication:
            output = F.adaptive_avg_pool3d(output, output_size=(1, 1, 1))
            output = torch.flatten(output, 1)
            output = self.fc_linear(output)

            return output
        else:
            return stage_features_list

   