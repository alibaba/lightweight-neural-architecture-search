# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import argparse
import ast
import logging
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modules import __all_blocks__, network_weight_stupid_init
from modules.qconv import QLinear


def parse_cmd_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--space_structure_txt', type=str, default=None)
    parser.add_argument('--space_structure_str', type=str, default=None)
    parser.add_argument('--space_dropout_channel', type=float, default=None)
    parser.add_argument('--space_dropout_layer', type=float, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


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
        raise ValueError('bad checkpoint to load %s' % (load_parameters_from))
    else:
        model.logger.debug('Zennas: loading params from '
                           + load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # print("\n#################################")
    # for name, paramets in model.named_parameters():
    # print(name, paramets.size(), paramets.flatten().cpu().detach().numpy()[0:5])
    #     if "conv_offset.weight" in name:
    #         state_dict[name] = state_dict[name.replace(".conv_offset", "")]
    model.load_state_dict(state_dict, strict=strict_load)

    # print("\n#################################")
    # for name, paramets in model.named_parameters():
    #     print(name, paramets.size(), paramets.flatten().cpu().detach().numpy()[0:5])

    return model


class CnnNet(nn.Module):

    def __init__(self,
                 num_classes=None,
                 structure_info=None,
                 structure_str=None,
                 structure_txt=None,
                 dropout_channel=None,
                 dropout_layer=None,
                 out_indices=(1, 2, 3, 4),
                 no_create=False,
                 classfication=False,
                 pretrained=None,
                 cfg=None,
                 logger=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.structure_info = structure_info
        self.structure_str = structure_str
        self.structure_txt = structure_txt
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer
        self.out_indices = out_indices
        self.classfication = classfication
        self.__all_blocks__ = __all_blocks__
        self.logger = logger or logging

        if self.structure_txt is not None:
            assert self.structure_str is None
            with open(self.structure_txt, 'r') as fid:
                self.structure_str = ''.join(
                    [x.strip() for x in fid.readlines()])

        if self.structure_str is not None:
            if self.structure_info is not None:
                self.logger.warning(
                    '--- Warning ! structure_info is not None when specifying structure_str !!!'
                )
            self.structure_info = ast.literal_eval(self.structure_str)
            assert isinstance(self.structure_info, list)

        if 'nbitsA' in self.structure_info[
                0] and 'nbitsW' in self.structure_info[0]:
            self.quant = True
        else:
            self.quant = False

        self.no_create = no_create

        self.block_list = nn.ModuleList()
        for block_structure_info in self.structure_info:
            the_block_class = self.__all_blocks__[
                block_structure_info['class']]
            the_block = the_block_class(
                block_structure_info, no_create=self.no_create, **kwargs)
            self.block_list.append(the_block)
        pass
    
        self.stage_idx = self.get_stage_idx()

        if self.classfication:  # output for the classfication task
            if self.quant:
                self.fc_linear = QLinear(
                    self.block_list[-1].out_channels,
                    self.num_classes,
                    bias=True,
                    nbits=8)
            else:
                self.fc_linear = nn.Linear(
                    self.block_list[-1].out_channels,
                    self.num_classes,
                    bias=True)

            network_weight_stupid_init(self.fc_linear)

    def init_weights(self, pretrained=None):
        """Initialize the weights of masternet.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_model(self, pretrained, strict=True)

        elif pretrained is None:
            pass
    
    def get_stage_idx(self,):
        stage_idx = []
        for idx, the_block in enumerate(self.block_list):

            if the_block.stride == 2 and 0 < idx < len(self.block_list):
                stage_idx.append(idx - 1)
            if idx == len(self.block_list) - 1:
                stage_idx.append(idx)
        return stage_idx

    def forward(self, x):
        # add different stages outputs for detection
        output = x
        if not self.classfication:
            stage_idx_output = [
                self.stage_idx[idx] for idx in self.out_indices
            ]
        stage_features_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if not self.classfication and idx in stage_idx_output:
                stage_features_list.append(output)

        if self.classfication:
            output = F.adaptive_avg_pool2d(output, output_size=(1, 1))
            output = torch.flatten(output, 1)
            output = self.fc_linear(output)

            if self.dropout_channel is not None:
                output = F.dropout(output, self.dropout_channel, self.training)

            return output
        else:
            return stage_features_list
