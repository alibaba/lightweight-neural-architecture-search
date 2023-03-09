# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import copy
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .blocks_basic import STD_BITS_LUT, BaseSuperBlock, ConvKXBN


class ResK1KXK1(nn.Module):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''
        :param structure_info: {
            'class': 'ResK1KXK1',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn': bottleneck_channels,
            'nbitsA': input activation quant nbits list(default=8),
            'nbitsW': weight quant nbits list(default=8),
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''

        super().__init__()

        #if 'class' in structure_info:
        #    assert structure_info['class'] == self.__class__.__name__

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.stride = 1 if 's' not in structure_info else structure_info['s']
        self.bottleneck_channels = structure_info['btn']
        assert self.stride == 1 or self.stride == 2
        if 'act' not in structure_info:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(structure_info['act'])
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer

        if 'force_resproj' in structure_info:
            self.force_resproj = structure_info['force_resproj']
        else:
            self.force_resproj = False

        if 'nbitsA' in structure_info and 'nbitsW' in structure_info:
            self.quant = True
            self.nbitsA = structure_info['nbitsA']
            self.nbitsW = structure_info['nbitsW']
            if len(self.nbitsA) != 3 or len(self.nbitsW) != 3:
                raise ValueError(
                    'nbitsA/W must has three elements in %s, not nbitsA %d or nbitsW %d'
                    % (self.__class__, len(self.nbitsA), len(self.nbitsW)))

        else:
            self.quant = False

        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        if 'p' in structure_info:
            self.padding = structure_info['p']
        else:
            self.padding = (self.kernel_size - 1) // 2

        self.model_size = 0.0
        self.flops = 0.0

        self.block_list = []

        conv1_info = {
            'in': self.in_channels,
            'out': self.bottleneck_channels,
            'k': 1,
            's': 1,
            'g': self.groups,
            'p': 0
        }
        conv2_info = {
            'in': self.bottleneck_channels,
            'out': self.bottleneck_channels,
            'k': self.kernel_size,
            's': self.stride,
            'g': self.groups,
            'p': self.padding
        }
        conv3_info = {
            'in': self.bottleneck_channels,
            'out': self.out_channels,
            'k': 1,
            's': 1,
            'g': self.groups,
            'p': 0
        }
        if self.quant:
            conv1_info = {
                **conv1_info,
                **{
                    'nbitsA': self.nbitsA[0],
                    'nbitsW': self.nbitsW[0]
                }
            }
            conv2_info = {
                **conv2_info,
                **{
                    'nbitsA': self.nbitsA[1],
                    'nbitsW': self.nbitsW[1]
                }
            }
            conv3_info = {
                **conv3_info,
                **{
                    'nbitsA': self.nbitsA[2],
                    'nbitsW': self.nbitsW[2]
                }
            }

        self.conv1 = ConvKXBN(conv1_info, no_create=no_create, **kwargs)
        self.conv2 = ConvKXBN(conv2_info, no_create=no_create, **kwargs)
        self.conv3 = ConvKXBN(conv3_info, no_create=no_create, **kwargs)

        # if self.no_create:
        #     pass
        # else:
        #     network_weight_stupid_bn_zero_init(self.conv3)

        self.block_list.append(self.conv1)
        self.block_list.append(self.conv2)
        self.block_list.append(self.conv3)

        self.model_size = self.model_size + self.conv1.get_model_size() + self.conv2.get_model_size() + \
            self.conv3.get_model_size()
        self.flops = self.flops + self.conv1.get_flops(1.0) \
            + self.conv2.get_flops(1.0) + self.conv3.get_flops(1.0 / self.stride) \
            + self.bottleneck_channels + self.bottleneck_channels \
            / self.stride**2 + self.out_channels / self.stride**2  # add relu flops

        # residual link
        if self.stride == 2:
            if self.no_create:
                pass
            else:
                self.residual_downsample = nn.AvgPool2d(
                    kernel_size=2, stride=2)
            self.flops = self.flops + self.in_channels
        else:
            if self.no_create:
                pass
            else:
                self.residual_downsample = nn.Identity()

        if self.in_channels != self.out_channels or self.force_resproj:
            if self.quant:
                self.residual_proj = ConvKXBN(
                    {
                        'in': self.in_channels,
                        'out': self.out_channels,
                        'k': 1,
                        's': 1,
                        'g': 1,
                        'p': 0,
                        'nbitsA': self.nbitsA[0],
                        'nbitsW': self.nbitsW[0]
                    },
                    no_create=no_create)
            else:
                self.residual_proj = ConvKXBN(
                    {
                        'in': self.in_channels,
                        'out': self.out_channels,
                        'k': 1,
                        's': 1,
                        'g': 1,
                        'p': 0
                    },
                    no_create=no_create)
            self.model_size = self.model_size + self.residual_proj.get_model_size(
            )
            self.flops = self.flops + self.residual_proj.get_flops(
                1.0 / self.stride) + self.out_channels / self.stride**2

            # if self.no_create:
            #     pass
            # else:
            #     network_weight_stupid_init(self.residual_proj)
        else:
            if self.no_create:
                pass
            else:
                self.residual_proj = nn.Identity()

    def forward(self, x, compute_reslink=True):
        reslink = self.residual_downsample(x)
        reslink = self.residual_proj(reslink)

        output = x
        output = self.conv1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        output = self.activation_function(output)
        output = self.conv2(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        output = self.activation_function(output)
        output = self.conv3(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        if self.dropout_layer is not None:
            if np.random.rand() <= self.dropout_layer:
                output = 0 * output + reslink
            else:
                output = output + reslink
        else:
            output = output + reslink

        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        output = self.activation_function(output)

        return output

    def get_model_size(self, return_list=False):
        if return_list:
            return self.conv1.get_model_size(
                return_list) + self.conv2.get_model_size(
                    return_list) + self.conv3.get_model_size(return_list)
        else:
            return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_layers(self):
        return 3

    def get_output_resolution(self, input_resolution):
        resolution = input_resolution
        for block in self.block_list:
            resolution = block.get_output_resolution(resolution)
        return resolution

    def get_params_for_trt(self, input_resolution):
        # generate the params for yukai's predictor
        params = []
        the_res = input_resolution
        for idx, block in enumerate(self.block_list):
            if self.residual_proj and idx == len(self.block_list) - 1:
                params_temp = block.get_params_for_trt(
                    the_res, elmtfused=1)  # if reslink, elmtfused=1
            else:
                params_temp = block.get_params_for_trt(the_res)
            the_res = block.get_output_resolution(the_res)
            params += params_temp
        if isinstance(self.residual_proj, ConvKXBN):
            params_temp = self.residual_proj.get_params_for_trt(the_res)
            params += params_temp

        return params

    def get_num_channels_list(self):
        return [
            self.bottleneck_channels, self.bottleneck_channels,
            self.out_channels
        ]

    def get_madnas_forward(self, **kwarg):
        if 'init_std' in kwarg and 'init_std_act' in kwarg and hasattr(
                self, 'nbitsA'):
            conv1_std = np.log(
                STD_BITS_LUT[kwarg['init_std_act']][self.nbitsA[0]]
                * STD_BITS_LUT[kwarg['init_std']][self.nbitsW[0]]) - np.log(
                    kwarg['init_std_act'])
            conv2_std = np.log(
                STD_BITS_LUT[kwarg['init_std_act']][self.nbitsA[1]]
                * STD_BITS_LUT[kwarg['init_std']][self.nbitsW[1]]) - np.log(
                    kwarg['init_std_act'])
            conv3_std = np.log(
                STD_BITS_LUT[kwarg['init_std_act']][self.nbitsA[2]]
                * STD_BITS_LUT[kwarg['init_std']][self.nbitsW[2]]) - np.log(
                    kwarg['init_std_act'])

            return [
                np.log(np.sqrt(self.in_channels)) + conv1_std + np.log(
                    np.sqrt(self.bottleneck_channels * self.kernel_size**2))
                + conv2_std + np.log(np.sqrt(self.bottleneck_channels))
                + conv3_std
            ]
        else:
            return [
                np.log(np.sqrt(self.in_channels)) + np.log(
                    np.sqrt(self.bottleneck_channels * self.kernel_size**2))
                + np.log(np.sqrt(self.bottleneck_channels))
            ]

    def get_max_feature_num(self, resolution):
        residual_featmap = resolution**2 * self.out_channels // (
            self.stride**2)
        if self.quant:
            residual_featmap = residual_featmap * self.nbitsA[0] / 8
        conv1_max_featmap = self.conv1.get_max_feature_num(
            resolution) + residual_featmap
        conv2_max_featmap = self.conv2.get_max_feature_num(
            resolution) + residual_featmap
        conv3_max_featmap = self.conv3.get_max_feature_num(resolution
                                                           // self.stride)
        max_featmap_list = [
            conv1_max_featmap, conv2_max_featmap, conv3_max_featmap
        ]

        return max_featmap_list


    def get_deepmad_forward(self, **kwarg):
        #alpha_config = {'alpha1':1, 'alpha2':1}
        return [
            np.log(np.sqrt(self.in_channels ** kwarg["alpha1"])) +
            np.log(np.sqrt(1 ** (2 * kwarg["alpha2"]))) +
            np.log(np.sqrt(self.bottleneck_channels ** kwarg["alpha1"])) +
            np.log(np.sqrt(self.kernel_size ** (2 * kwarg["alpha2"]))) +
            np.log(np.sqrt(self.bottleneck_channels ** kwarg["alpha1"])) +
            np.log(np.sqrt(1 ** (2 * kwarg["alpha2"])))
        ]

    def get_width(self, ):
        # c*1^2 * ck^2 * c*1^2 = c*c*c*k^2
        return [
            self.in_channels * 1 ** 2 *
            self.bottleneck_channels * self.kernel_size ** 2 *
            self.bottleneck_channels * 1 ** 2
        ]


class SuperResK1KXK1(BaseSuperBlock):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'SuperResK1KXK1',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
        }
        :param NAS_mode:
        '''
        structure_info['inner_class'] = 'ResK1KXK1'
        super().__init__(
            structure_info=structure_info,
            no_create=no_create,
            inner_class=ResK1KXK1,
            dropout_channel=dropout_channel,
            dropout_layer=dropout_layer,
            **kwargs)


__module_blocks__ = {
    'ResK1KXK1': ResK1KXK1,
    'SuperResK1KXK1': SuperResK1KXK1,
}
