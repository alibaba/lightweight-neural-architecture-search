# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ast, argparse

from .blocks import __all_blocks__, network_weight_stupid_init


def parse_cmd_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--space_structure_txt', type=str, default=None)
    parser.add_argument('--space_structure_str', type=str, default=None)
    parser.add_argument('--space_block_module', type=str, default=None)
    parser.add_argument('--space_dropout_channel', type=float, default=None)
    parser.add_argument('--space_dropout_layer', type=float, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


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


def load_model(model, load_parameters_from, strict_load=False, map_location=torch.device('cpu'), **kwargs):
    if not os.path.isfile(load_parameters_from):
        raise ValueError("bad checkpoint to load %s"%(load_parameters_from))
    else:
        print('Zennas: loading params from ' + load_parameters_from)
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


class MasterNet(nn.Module):
    def __init__(self, num_classes=None, structure_info=None, structure_str=None, structure_txt=None,
                 block_module=None, dropout_channel=None, dropout_layer=None, out_indices=(1, 2, 3, 4),
                 classfication=False, argv=None, no_create=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.structure_info = structure_info
        self.structure_str = structure_str
        self.structure_txt = structure_txt
        self.block_module = block_module
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer
        self.out_indices = out_indices
        self.classfication = classfication
        self.__all_blocks__ = __all_blocks__

        if argv is not None:
            assert structure_info is None and structure_str is None and structure_txt is None
            args = parse_cmd_args(argv)
            self.structure_str = args.space_structure_str
            self.structure_txt = args.space_structure_txt
            self.dropout_channel = args.space_dropout_channel
            self.dropout_layer = args.space_dropout_layer
            self.block_module = load_py_module_from_path(args.space_block_module)

        if self.structure_txt is not None:
            assert self.structure_str is None
            with open(self.structure_txt, 'r') as fid:
                self.structure_str = ''.join([x.strip() for x in fid.readlines()])
        pass

        if self.structure_str is not None:
            if self.structure_info is not None:
                print('--- Warning ! structure_info is not None when specifying structure_str !!!')
            self.structure_info = ast.literal_eval(self.structure_str)
            assert isinstance(self.structure_info, list)

        self.no_create = no_create

        if isinstance(self.block_module, str):
            self.block_module = load_py_module_from_path(self.block_module)
            self.__all_blocks__.update(self.block_module.__module_blocks__)


        self.block_list = nn.ModuleList()
        for block_structure_info in self.structure_info:
            the_block_class = self.__all_blocks__[block_structure_info['class']]
            the_block = the_block_class(block_structure_info, no_create=self.no_create, **kwargs)
            self.block_list.append(the_block)
        pass

        if self.classfication: # output for the classfication task
            self.fc_linear = nn.Linear(self.block_list[-1].out_channels, self.num_classes, bias=True)
            
            network_weight_stupid_init(self.fc_linear)

        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels = self.get_stage_info()
        # set dropout rate
        L = self.get_num_layers()
        current_depth = 0
        for block in self.block_list:
            current_depth += block.get_num_layers()
            if self.dropout_channel is not None:
                block.dropout_channel = self.dropout_channel * current_depth / L
            if self.dropout_layer is not None:
                block.dropout_layer = self.dropout_layer * current_depth / L


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
        if not self.classfication: stage_idx_output = [self.stage_idx[idx] for idx in self.out_indices]
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


    def forward_inner_layer_features(self, x):
        inner_layer_features = []
        output = x
        for block in self.block_list:
            if hasattr(block, 'forward_inner_layer_features'):
                output, tmp_inner_layer_features = block.forward_inner_layer_features(output)
                inner_layer_features += tmp_inner_layer_features
                inner_layer_features.append(output)
            else:
                output = block(output)
                inner_layer_features.append(output)
        output = F.adaptive_avg_pool2d(output, output_size=(1, 1))
        output = torch.flatten(output, 1)
        output = self.fc_linear(output)

        return output, inner_layer_features


    def get_model_size(self):
        model_size = 0
        for block in self.block_list:
            model_size += block.get_model_size()

        if self.classfication: model_size += self.block_list[-1].out_channels * self.num_classes + self.num_classes  # for fc_linear

        return model_size


    def get_flops(self, resolution):
        flops = 0.0
        the_res = resolution
        for block in self.block_list:
            flops += block.get_flops(the_res)
            the_res /= block.stride

        if self.classfication: flops += self.block_list[-1].out_channels * self.num_classes  # for fc_linear

        return flops


    def get_num_layers(self):
        n = 0
        for block in self.block_list:
            n += block.get_num_layers()
        return n


    def get_num_stages(self):

        num_stages = 0
        for the_block in self.block_list:
            if the_block.stride == 2:
                num_stages += 1
            elif not the_block.stride == 1:
                raise ValueError("stride must equals to 1 or 2, not %d"%(the_block.stride))

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


    def get_stage_info(self,):
        stage_idx = []
        stage_channels = []
        stage_block_num = []
        stage_layer_num = []

        channel_num = 0
        block_num = 0
        layer_num = 0
        for idx, the_block in enumerate(self.block_list):

            if the_block.stride == 2 and 0<idx<len(self.block_list):
                stage_idx.append(idx-1)
                stage_channels.append(channel_num)
                stage_block_num.append(block_num)
                stage_layer_num.append(layer_num)

            block_num += the_block.get_block_num()
            channel_num = the_block.out_channels
            layer_num += the_block.get_num_layers()

            if idx==len(self.block_list)-1:
                stage_idx.append(idx)
                stage_channels.append(channel_num)
                stage_block_num.append(block_num)
                stage_layer_num.append(layer_num)

        return stage_idx, stage_block_num, stage_layer_num, stage_channels


    def entropy_forward_pre_GAP(self, x, skip_relu=True, skip_bn=True, **kwarg):
        # BN must be removed when calculated the entropy, block is the small unit
        output = x
        block_std_list = []
        stage_features_list = []
        for idx, the_block in enumerate(self.block_list):
            output, output_std_list_plain = the_block.entropy_forward(output, skip_relu=skip_relu, skip_bn=skip_bn, **kwarg)
            if idx in self.stage_idx:
                stage_features_list.append(output)
            block_std_list += output_std_list_plain
        return stage_features_list, block_std_list