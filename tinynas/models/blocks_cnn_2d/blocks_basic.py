# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import copy
import os
import sys

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .qconv import QConv2d

STD_BITS_LUT = {
    1: {
        2: 1.0089193649965897,
        3: 1.0408034134404924,
        4: 1.0408329944621844,
        5: 1.0408329944621846,
        6: 1.0408329944621846,
        7: 1.0408329944621846,
        8: 1.0408329944621846
    },
    2: {
        2: 1.470493611987268,
        3: 1.9441392428674842,
        4: 2.020625612100753,
        5: 2.020725942163689,
        6: 2.0207259421636903,
        7: 2.0207259421636903,
        8: 2.0207259421636903
    },
    3: {
        2: 1.6488958630350534,
        3: 2.538810044255356,
        4: 2.9936671208664336,
        5: 3.0138566423355098,
        6: 3.013856886670854,
        7: 3.013856886670854,
        8: 3.013856886670854
    },
    4: {
        2: 1.7388228634059972,
        3: 2.8903940182483057,
        4: 3.8504712566997665,
        5: 4.010172973403067,
        6: 4.010403138505318,
        7: 4.010403138505321,
        8: 4.010403138505321
    },
    5: {
        2: 1.7924496624207404,
        3: 3.112192524683381,
        4: 4.528190206368193,
        5: 5.0020480504267,
        6: 5.008326399730913,
        7: 5.008326400438906,
        8: 5.008326400438906
    },
    6: {
        2: 1.8279355957465278,
        3: 3.2624595365613334,
        4: 5.046533952705038,
        5: 5.965323558459142,
        6: 6.006939888476478,
        7: 6.006940430313366,
        8: 6.006940430313366
    },
    8: {
        2: 1.8718895621820542,
        3: 3.45101648516446,
        4: 5.755434102119865,
        5: 7.6819204526202745,
        6: 8.004730132702088,
        7: 8.005206639015208,
        8: 8.005206639015217
    },
    12: {
        2: 1.915289815677668,
        3: 3.6382645717164728,
        4: 6.507222754600756,
        5: 10.077501629698418,
        6: 11.919610277123745,
        7: 12.003470607624022,
        8: 12.003471720020558
    },
    16: {
        2: 1.936748155491386,
        3: 3.7306952212335203,
        4: 6.888695319518398,
        5: 11.498183425186014,
        6: 15.354315080598655,
        7: 16.001642665300874,
        8: 16.00260395477351
    }
}


def network_weight_stupid_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(
                        k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def network_weight_stupid_bn_zero_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(
                        k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.zeros_(m.weight)  # NOTE: BN is initialized to Zero
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def network_weight_bn_zero_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                device = m.weight.device
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(
                        k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net


class Swish(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def swish(x: Tensor) -> Tensor:
    return x * F.sigmoid(x)


def get_activation(name='relu'):
    if name == 'sigmoid':
        module = F.sigmoid
    elif name == 'relu':
        module = F.relu
    elif name == 'relu6':
        module = F.relu6
    elif name == 'swish':
        module = swish
    elif name == 'learkyrelu':
        module = F.leaky_relu
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class ConvKXBN(nn.Module):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'ConvKX',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'nbitsA': input activation quant nbits (default=8),
            'nbitsW': weight quant nbits (default=8),
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
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer

        if 'nbitsA' in structure_info and 'nbitsW' in structure_info:
            self.quant = True
            self.nbitsA = structure_info['nbitsA']
            self.nbitsW = structure_info['nbitsW']
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

        if self.no_create:
            self.block = None
        else:
            if self.quant:
                self.conv1 = QConv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    groups=self.groups,
                    bias=False,
                    nbitsA=self.nbitsA,
                    nbitsW=self.nbitsW,
                    quan_type='lsq',
                    positive=False,
                    **kwargs)
            else:
                self.conv1 = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    padding=self.padding,
                    groups=self.groups,
                    bias=False)
            self.bn1 = nn.BatchNorm2d(self.out_channels)

        if self.quant:
            self.model_size = self.model_size + self.in_channels * self.out_channels * \
                self.kernel_size ** 2 / self.groups * self.nbitsW / 8 + 2 * self.out_channels
            self.flops = self.flops + self.in_channels * self.out_channels * self.kernel_size**2 / self.stride**2 / \
                self.groups * self.nbitsA * self.nbitsW / 8 / 8 + 2 * self.out_channels / self.stride**2
        else:
            self.model_size = self.model_size + self.in_channels * self.out_channels * \
                self.kernel_size**2 / self.groups + 2 * self.out_channels
            self.flops = self.flops + self.in_channels * self.out_channels * self.kernel_size**2 / \
                self.stride**2 / self.groups + 2 * self.out_channels / self.stride**2

    def forward(self, x, skip_bn=False):
        output = self.conv1(x)
        if not skip_bn:
            output = self.bn1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        return output

    def get_model_size(self, return_list=False):
        if return_list:
            return [self.model_size]
        else:
            return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_params_for_trt(self, input_resolution, elmtfused=0):
        # generate the params for yukai's predictor
        if self.groups == 1:
            return [('Regular', self.stride, elmtfused, self.kernel_size, 1,
                     self.in_channels, input_resolution, self.out_channels)]
        elif self.groups == self.out_channels:
            return [('Depthwise', self.stride, elmtfused, self.kernel_size, 1,
                     self.in_channels, input_resolution, self.out_channels)]
        else:
            raise ValueError(
                'Conv or DepthWise are supported in predictor, not Group Conv.'
            )

    def get_layers(self):
        return 1

    def get_num_channels_list(self):
        return [self.out_channels]

    def get_max_feature_num(self, resolution, nbitsA_out=8):

        nbitsA_in = self.nbitsA if self.quant else 8
        if self.groups == 1:
            max_feature = resolution**2 * self.in_channels * nbitsA_in / 8 + resolution**2 * self.out_channels // (
                self.stride**2) * nbitsA_out / 8
        elif self.groups == self.out_channels:
            max_feature = resolution**2 * self.in_channels * nbitsA_in / 8 + resolution**2 // (
                self.stride**2) * nbitsA_out / 8  # TinyEngine-style
        else:
            raise ValueError(
                'Conv or DepthWise are supported in max_feature_num, not Group Conv.'
            )

        return max_feature


class ConvKXBNRELU(ConvKXBN):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'ConvKXBNRELU',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'nbitsA': input activation quant nbits (default=8),
            'nbitsW': weight quant nbits (default=8),
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''
        super().__init__(
            structure_info=structure_info,
            no_create=no_create,
            dropout_channel=dropout_channel,
            dropout_layer=dropout_layer,
            **kwargs)
        if 'act' not in structure_info:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(structure_info['act'])
        self.flops = self.flops + self.out_channels / self.stride**2  # add relu flops

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        return self.activation_function(output)

    def get_block_num(self):
        return 1

    def entropy_forward(self, x, skip_relu=True, skip_bn=True, **kwarg):
        output = self.conv1(x)
        output_std_list = []
        if not skip_bn:
            output = self.bn1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        if not skip_relu:
            output = self.activation_function(output)
        if 'init_std_act' in kwarg and hasattr(self, 'nbitsA'):
            output_std_list.append(output.std() / kwarg['init_std_act'])
            output = output / (output.std() / kwarg['init_std_act'])
        else:
            output_std_list.append(output.std())
            output = output / (output.std())
        return output, output_std_list

    def get_madnas_forward(self, **kwarg):
        if 'init_std' in kwarg and 'init_std_act' in kwarg and hasattr(
                self, 'nbitsA'):
            conv_std = np.log(
                STD_BITS_LUT[kwarg['init_std_act']][self.nbitsA]
                * STD_BITS_LUT[kwarg['init_std']][self.nbitsW]) - np.log(
                    kwarg['init_std_act'])
            return [
                np.log(np.sqrt(self.in_channels * self.kernel_size**2))
                + conv_std
            ]
        else:
            return [np.log(np.sqrt(self.in_channels * self.kernel_size**2))]


    def get_deepmad_forward(self, **kwarg):
        #alpha_config = {'alpha1':1, 'alpha2':1}
        return [
            np.log(np.sqrt((self.in_channels ** kwarg["alpha1"])
            *  (self.kernel_size ** (2 * kwarg["alpha2"]))))
        ]

    def get_width(self, ):
        return [
            self.in_channels * self.kernel_size ** 2
        ]

class SE(nn.Module):
    def __init__(self, structure_info, no_create=False,
                 **kwargs):
        '''
        :param structure_info: {
            'class': 'SE',
            'in': in_channels,
            'out': out_channels,
            'se_ratio': se_ratio (default=0.25 * in_channels),
        }
        :param NAS_mode:
        '''

        super().__init__()

        if 'class' in structure_info:
            assert structure_info['class'] == self.__class__.__name__

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.no_create = no_create

        if 'se_ratio' in structure_info:
            self.se_ratio = structure_info['se_ratio']
        else:
            self.se_ratio = 0.25

        self.btn_channels = max(1, round(self.in_channels * self.se_ratio) // 8) * 8

        self.model_size = 0.0
        self.flops = 0.0

        conv1_info = {'in': self.in_channels, 'out': self.btn_channels, 'k': 1}
        conv2_info = {'in': self.btn_channels, 'out': self.out_channels, 'k': 1}

        self.conv1 = ConvKXBN(conv1_info, no_create=no_create)
        self.conv2 = ConvKXBN(conv2_info, no_create=no_create)

        self.model_size = self.model_size + self.conv1.get_model_size() + self.conv2.get_model_size()
        self.flops = self.flops + self.conv1.get_flops(resolution=1) + self.conv2.get_flops(resolution=1) \
                     + self.btn_channels + self.out_channels
        self.activation_function = nn.SiLU()  # F.relu
        # self.activation_function = nn.ReLU()  # F.relu

    def forward(self, x):
        output = F.adaptive_avg_pool2d(x, output_size=(1,1))
        output = self.activation_function(self.conv1(output))
        output = self.conv2(output)
        output = F.sigmoid(output)
        output = output * x

        return output

    def get_model_size(self):
        return self.model_size

    def get_flops(self, resolution):
        return self.flops

    def get_num_layers(self):
        return 0

    def get_num_channels_list(self):
        return [self.btn_channels, self.out_channels]

    def get_output_resolution(self, input_resolution):
        # output_resolution = self.conv1.get_output_resolution(input_resolution)
        # output_resolution = self.conv2.get_output_resolution(output_resolution)
        # return output_resolution
        return 1

    def get_params_for_trt(self, input_resolution, elmtfused=0):
        # generate the params for yukai's predictor
        params = []
        params_temp = self.conv1.get_params_for_trt(input_resolution)
        params += params_temp
        output_resolution = self.conv1.get_output_resolution(input_resolution)
        params_temp = self.conv2.get_params_for_trt(output_resolution)
        params += params_temp
        return params

    # def get_log_mad_score(self, **kwarg):
    #     return [np.log(np.sqrt(self.in_channels ** kwarg["alpha1"])) + \
    #             np.log(np.sqrt(1 ** (2 * kwarg["alpha2"]))) + \
    #             np.log(np.sqrt(self.btn_channels ** kwarg["alpha1"])) + \
    #             np.log(np.sqrt(1 ** (2 * kwarg["alpha2"])))]

class BaseSuperBlock(nn.Module):

    def __init__(self,
                 structure_info,
                 no_create=False,
                 inner_class=None,
                 dropout_channel=None,
                 dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'BaseSuperBlock',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
            'inner_class': inner_class,
            'force_resproj_skip': force_resproj_skip (default=4),
            'nbitsA': input activation quant nbits list(default=8),
            'nbitsW': weight quant nbits list(default=8),
        }
        :param NAS_mode:
        '''

        super().__init__()

        #if 'class' in structure_info:
        #    assert structure_info['class'] == self.__class__.__name__

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        # self.kernel_size = structure_info['k']
        self.stride = 1 if 's' not in structure_info else structure_info['s']
        # if 'btn' in structure_info:
        #     self.bottleneck_channels = structure_info['btn']
        # else:
        #     self.bottleneck_channels = None
        self.inner_class_name = structure_info['inner_class']
        self.inner_class = inner_class
        self.num_inner_layers = structure_info['L']
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer

        assert self.stride == 1 or self.stride == 2

        if 'nbitsA' in structure_info and 'nbitsW' in structure_info:
            self.quant = True
            self.nbitsA = structure_info['nbitsA']
            self.nbitsW = structure_info['nbitsW']
            self.inner_layers = len(
                structure_info['nbitsA']) // self.num_inner_layers
        else:
            self.quant = False

        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        # if 'p' in structure_info:
        #     self.padding = structure_info['p']
        # else:
        #     self.padding = (self.kernel_size - 1) // 2

        if 'force_resproj_skip' in structure_info:
            self.force_resproj_skip = structure_info['force_resproj_skip']
        else:
            self.force_resproj_skip = 4

        self.model_size = 0.0
        self.flops = 0.0

        self.block_list = nn.ModuleList()

        current_res = 1.0
        for block_id in range(self.num_inner_layers):
            if block_id == 0:
                in_channels = self.in_channels
                out_channels = self.out_channels
                stride = self.stride
                # True for K1KXK1, False for others
                force_resproj = True if structure_info[
                    'inner_class'] == 'ResK1KXK1' else False
            elif block_id % self.force_resproj_skip == 0:
                in_channels = self.out_channels
                out_channels = self.out_channels
                stride = 1
                force_resproj = False
            else:
                in_channels = self.out_channels
                out_channels = self.out_channels
                stride = 1
                force_resproj = False

            inner_structure_info = copy.deepcopy(structure_info)
            inner_structure_info['in'] = in_channels
            inner_structure_info['out'] = out_channels
            inner_structure_info['s'] = stride
            inner_structure_info['force_resproj'] = force_resproj

            inner_structure_info['class'] = inner_structure_info['inner_class']
            if self.quant:
                inner_structure_info['nbitsA'] = structure_info[
                    'nbitsA'][block_id * self.inner_layers:(block_id + 1)
                              * self.inner_layers]
                inner_structure_info['nbitsW'] = structure_info[
                    'nbitsW'][block_id * self.inner_layers:(block_id + 1)
                              * self.inner_layers]

            the_block = self.inner_class(
                structure_info=inner_structure_info,
                no_create=no_create,
                dropout_channel=self.dropout_channel,
                dropout_layer=self.dropout_layer,
                **kwargs)

            self.block_list.append(the_block)
            self.model_size = self.model_size + the_block.get_model_size()
            self.flops = self.flops + the_block.get_flops(current_res)
            current_res /= stride

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)

        return output

    def forward_inner_layer_features(self, x):
        inner_layer_features = []
        output = x
        for block_id, block in enumerate(self.block_list):
            output = block(output)
            if block_id <= len(self.block_list) - 2 and block_id % 4 == 3:
                inner_layer_features.append(output)

        return output, inner_layer_features

    def get_model_size(self, return_list=False):
        if return_list:
            model_size_list = []
            for block_id, block in enumerate(self.block_list):
                model_size_list += block.get_model_size(return_list)
            return model_size_list
        return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_layers(self):
        L = 0
        for block in self.block_list:
            L = L + block.get_layers()
        return L

    def get_block_num(self):
        return len(self.block_list)

    def get_output_resolution(self, input_resolution):
        resolution = input_resolution
        for block in self.block_list:
            resolution = block.get_output_resolution(resolution)
        return resolution

    def get_params_for_trt(self, input_resolution):
        # generate the params for yukai's predictor
        params = []
        the_res = input_resolution
        for block in self.block_list:
            params_temp = block.get_params_for_trt(the_res)
            the_res = block.get_output_resolution(the_res)
            params += params_temp
        return params

    def entropy_forward(self, x, skip_relu=True, skip_bn=True, **kwarg):
        output = x
        output_std_list = []
        for the_block in self.block_list:
            output, output_std_list_plain = the_block.entropy_forward(
                output, skip_relu=skip_relu, skip_bn=skip_bn, **kwarg)
            output_std_list += output_std_list_plain
        return output, output_std_list

    def get_num_channels_list(self):
        num_channels_list = []
        for block in self.block_list:
            num_channels_list += block.get_num_channels_list()

        return num_channels_list

    def get_madnas_forward(self, **kwarg):
        output_std_list_plain = []
        for block in self.block_list:
            output_std_list_plain += block.get_madnas_forward(**kwarg)
        return output_std_list_plain

    def sym_get_model_size(self):
        sym_model_size = self.block_list[0].get_model_size(
        ) + self.block_list[1].get_model_size() * (
            self.num_inner_layers - 1)
        return sym_model_size

    def sym_get_flops(self, resolution):
        sym_flops = self.block_list[0].get_flops(resolution)
        resolution = resolution / self.block_list[0].stride
        sym_flops = sym_flops + self.block_list[1].get_flops(resolution) * (
            self.num_inner_layers - 1)
        return sym_flops

    def get_max_feature_num(self, resolution, nbitsA_out=8):
        the_res = resolution
        max_featmap_list = []

        for idx, the_block in enumerate(self.block_list, 0):
            if self.quant:
                if idx < len(self.block_list) - 1:
                    nbitsA_next = self.block_list[idx + 1].nbitsA[0]
                else:
                    nbitsA_next = nbitsA_out
            else:
                nbitsA_next = 8
            temp_featmap_list = the_block.get_max_feature_num(
                the_res, nbitsA_out=nbitsA_next)
            the_res = the_block.get_output_resolution(the_res)
            max_featmap_list += temp_featmap_list

        return max_featmap_list

    def get_deepmad_forward(self, **kwarg):
        output_std_list_plain = []
        for block in self.block_list:
            output_std_list_plain += block.get_deepmad_forward(**kwarg)
        return output_std_list_plain

    def get_width(self, ):
        output_width_list = []
        for block in self.block_list:
            output_width_list += block.get_width()
        return output_width_list


__module_blocks__ = {
    'SE': SE,
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
}
