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

from .blocks_cnn_2d import __all_blocks__, network_weight_stupid_init
from .blocks_cnn_2d.qconv import QLinear
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
        raise ValueError('bad checkpoint to load %s' % (load_parameters_from))
    else:
        model.logger.debug('Zennas: loading params from '
                           + load_parameters_from)
    checkpoint = torch.load(load_parameters_from, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict_load)

    return model

@MODELS.register_module(module_name = 'CnnNet')
class CnnNet(Model):

    def __init__(self,
                 structure_info,
                 dropout_channel = None,
                 dropout_layer = None,
                 out_indices=(1, 2, 3, 4),
                 no_create=False,
                 pretrained=None,
                 logger=None,
                 **kwargs):
        self.out_indices = out_indices
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer
        self.__all_blocks__ = __all_blocks__
        assert structure_info, 'Must set structure_info'

        super().__init__(structure_info, logger)
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

        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels, _ = self.get_stage_info(
        )
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
            load_model(self, pretrained, strict=True)

        elif pretrained is None:
            pass

    def forward(self, x):
        # add different stages outputs for detection
        output = x
        stage_idx_output = [
                self.stage_idx[idx] for idx in self.out_indices
            ]
        stage_features_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in stage_idx_output:
                stage_features_list.append(output)

        return stage_features_list

    def forward_inner_layer_features(self, x):
        inner_layer_features = []
        output = x
        for block in self.block_list:
            if hasattr(block, 'forward_inner_layer_features'):
                output, tmp_inner_layer_features = block.forward_inner_layer_features(
                    output)
                inner_layer_features += tmp_inner_layer_features
                inner_layer_features.append(output)
            else:
                output = block(output)
                inner_layer_features.append(output)
        output = F.adaptive_avg_pool2d(output, output_size=(1, 1))
        output = torch.flatten(output, 1)
        output = self.fc_linear(output)

        return output, inner_layer_features

    def get_model_size(self, return_list=False):
        model_size = 0
        model_size_list = []
        for block in self.block_list:
            model_size += block.get_model_size()
            model_size_list += block.get_model_size(return_list=True)

        if return_list:
            return model_size_list
        else:
            return model_size

    def get_flops(self, resolution):
        flops = 0.0
        the_res = resolution
        for block in self.block_list:
            flops += block.get_flops(the_res)
            the_res /= block.stride

        return flops

    def get_latency(self, latency_func):
        return latency_func(self)

    def get_layers(self):
        n = 0
        for block in self.block_list:
            n += block.get_layers()
        return n

    def get_stages(self):

        num_stages = 0
        for the_block in self.block_list:
            if the_block.stride == 2:
                num_stages += 1
            elif not the_block.stride == 1:
                raise ValueError('stride must equals to 1 or 2, not %d' %
                                 (the_block.stride))

        return num_stages

    def get_params_for_trt(self, input_resolution):
        # generate the params for yukai's predictor
        params = []
        the_res = input_resolution
        for block in self.block_list:
            params_temp = block.get_params_for_trt(the_res)
            the_res = block.get_output_resolution(the_res)
            params += params_temp
        return params

    def get_stage_info(self, resolution=224):
        stage_idx = []
        stage_channels = []
        stage_block_num = []
        stage_layer_num = []

        stage_feature_map_size = []
        feature_map_size = resolution  # use the input size as initialization

        channel_num = 0
        block_num = 0
        layer_num = 0
        for idx, the_block in enumerate(self.block_list):

            if the_block.stride == 2 and 0 < idx < len(self.block_list):
                stage_idx.append(idx - 1)
                stage_channels.append(channel_num)
                stage_block_num.append(block_num)
                stage_layer_num.append(layer_num)
                stage_feature_map_size.append(feature_map_size)

            block_num += the_block.get_block_num()
            channel_num = the_block.out_channels
            layer_num += the_block.get_layers()
            feature_map_size = the_block.get_output_resolution(feature_map_size)

            if idx == len(self.block_list) - 1:
                stage_idx.append(idx)
                stage_channels.append(channel_num)
                stage_block_num.append(block_num)
                stage_layer_num.append(layer_num)
                stage_feature_map_size.append(feature_map_size)

        return stage_idx, stage_block_num, stage_layer_num, stage_channels, stage_feature_map_size


    def madnas_forward_pre_GAP(self, **kwarg):
        # BN must be removed when calculated the entropy, block is the small unit
        block_std_list = []
        for idx, the_block in enumerate(self.block_list):
            output_std_list_plain = the_block.get_madnas_forward(**kwarg)
            block_std_list += output_std_list_plain
        return block_std_list

    def deepmad_forward_pre_GAP(self, **kwarg):
        network_std_list= []
        for idx, the_block in enumerate(self.block_list):
            one_std_list = the_block.get_deepmad_forward(**kwarg)
            network_std_list += one_std_list
        return network_std_list

    def get_efficient_score(self, **kwarg):
        log_width_list = []
        network_block_width_list = []
        for idx, the_block in enumerate(self.block_list):
            one_width = the_block.get_width()
            network_block_width_list += one_width

        for idx in range(len(self.stage_layer_num)):

            rho_width = 0.0
            for idx1 in range(self.stage_block_num[idx]):
                rho_width += np.log(network_block_width_list[idx1])
            log_width_list.append(rho_width)

        log_width_arr = np.array(log_width_list)
        depth_arr = np.array(self.stage_layer_num)
        effective_score_arr = np.exp(np.log(depth_arr) - log_width_arr / depth_arr)
        return float(np.max(effective_score_arr))

    def get_max_feature_num(self, resolution, nbitsA_out=8):
        the_res = resolution
        max_featmap_list = []

        for idx, the_block in enumerate(self.block_list, 0):
            if self.quant:
                if idx < len(self.block_list) - 1:
                    if type(self.block_list[idx + 1].nbitsA) == list:
                        nbitsA_next = self.block_list[idx + 1].nbitsA[0]
                    else:
                        nbitsA_next = self.block_list[idx + 1].nbitsA
                else:
                    nbitsA_next = nbitsA_out
            else:
                nbitsA_next = 8
            temp_featmap_list = the_block.get_max_feature_num(
                the_res, nbitsA_out=nbitsA_next)
            the_res = the_block.get_output_resolution(the_res)
            if isinstance(temp_featmap_list, list):
                max_featmap_list += temp_featmap_list
            else:
                max_featmap_list.append(temp_featmap_list)

        return max_featmap_list

    def get_max_feature(self, resolution):
        return np.max(self.get_max_feature_num(resolution))

    def build(self, structure_info):
        if structure_info is None:
            return self
        return CnnNet(
            structure_info=structure_info,
            dropout_channel=self.dropout_channel,
            dropout_layer=self.dropout_layer,
            out_indices=self.out_indices,
            no_create=False,
        )
