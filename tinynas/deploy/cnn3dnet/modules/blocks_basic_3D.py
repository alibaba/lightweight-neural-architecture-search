import os,sys
import copy
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


def network_weight_stupid_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                continue

    return net

def network_weight_stupid_bn_zero_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.zeros_(m.weight)  # NOTE: BN is initialized to Zero
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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


def get_activation(name="relu"):
    if name == "sigmoid":
        module = F.sigmoid
    elif name == "relu":
        module = F.relu
    elif name == "relu6":
        module = F.relu6
    elif name == "swish":
        module = swish
    elif name == "learkyrelu":
        module = F.leaky_relu
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Conv3DKXBN(nn.Module):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'ConvKX',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'kt': kernel_3d,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''

        super().__init__()

        if 'class' in structure_info:
            assert structure_info['class'] == self.__class__.__name__

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.kernel_3d = structure_info['kt']
        self.stride = 1 if 's' not in structure_info else structure_info['s']
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer


        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        if 'p' in structure_info:
            self.padding = structure_info['p']
        else:
            padding_3d = (self.kernel_3d - 1) // 2
            padding_2d = (self.kernel_size - 1) // 2
            self.padding = (padding_3d, padding_2d, padding_2d)

        self.model_size = 0.0
        self.flops = 0.0

        if self.no_create:
            self.block = None
        else:
            kernel_tuple = (self.kernel_3d, self.kernel_size, self.kernel_size)
            stride_tuple = (1, self.stride, self.stride)
            self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, kernel_tuple, stride_tuple,
                          padding=self.padding, groups=self.groups, bias=False)
            self.bn1 = nn.BatchNorm3d(self.out_channels)

        self.model_size = self.model_size + self.in_channels * self.out_channels * self.kernel_3d * self.kernel_size**2 / self.groups \
                           + 2 * self.out_channels
        self.flops = self.flops + self.in_channels * self.out_channels * self.kernel_3d * self.kernel_size**2 / self.stride**2 / self.groups \
                      + 2 * self.out_channels / self.stride**2

    def forward(self, x, skip_bn=False):
        output = self.conv1(x)
        if not skip_bn: output = self.bn1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        return output

    def get_model_size(self):
        return self.model_size

    def get_flops(self, resolution):
        return self.flops * resolution**2

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_params_for_trt(self, input_resolution, elmtfused=0):
        # generate the params for yukai's predictor
        if self.groups == 1:
            return [("Regular", self.stride, elmtfused, self.kernel_size, 1, self.in_channels, input_resolution, self.out_channels)]
        elif self.groups == self.out_channels:
            return [("Depthwise", self.stride, elmtfused, self.kernel_size, 1, self.in_channels, input_resolution, self.out_channels)]
        else:
            raise ValueError('Conv or DepthWise are supported in predictor, not Group Conv.')

    def get_layers(self):
        return 1

    def get_num_channels_list(self):
        return [self.out_channels]

    def get_stentr_forward(self, **kwarg):
        alpha_w = get_alpha_w(kwarg["resolution"], kwarg["frames"], self.kernel_3d, self.kernel_size)
        _score = np.log(np.sqrt(self.in_channels * self.kernel_3d * self.kernel_size**2 * alpha_w))
        return [_score]

def get_alpha_w(Res, frame, Kt, k, mode=1):
    # Res = Res*2
    if k==1:
        return 1.0
    coffe1 = np.array([frame, Res, Res])
    coffe2 = np.array([Kt, k, k])
    if mode == 1:
        coffe1_stand = coffe1/np.linalg.norm(coffe1)
        coffe2_stand = coffe2/np.linalg.norm(coffe2)
        alpha_w = (-np.log(1-np.sum(coffe1_stand*coffe2_stand)))
    return alpha_w


class Conv3DKXBNRELU(Conv3DKXBN):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'Conv3DKXBNRELU',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'kt': kernel_3d,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''
        super().__init__(structure_info=structure_info, no_create=no_create,
                         dropout_channel=dropout_channel, dropout_layer=dropout_layer,
                         **kwargs)
        if "act" not in structure_info:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(structure_info['act'])
        self.flops = self.flops + self.out_channels / self.stride ** 2  # add relu flops

    def forward(self, x):
        print(x.size())
        output = self.conv1(x)
        output = self.bn1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        return self.activation_function(output)

    def get_block_num(self):
        return 1

    def entropy_forward(self, x, skip_relu=True, skip_bn=True):
        output = self.conv1(x)
        output_std_list = []
        if not skip_bn: output = self.bn1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        if not skip_relu: output = self.activation_function(output)
        output_std_list.append(output.std())
        output = output/output.std()
        return output, output_std_list


class SE3D(nn.Module):
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

        self.conv1 = Conv3DKXBN(structure_info={
                'in': self.in_channels,
                'out': self.btn_channels,
                'k': 1, 'kt': 1}, no_create=no_create)

        self.conv2 = Conv3DKXBN(structure_info={
            'in': self.btn_channels,
            'out': self.out_channels,
            'k': 1, 'kt': 1}, no_create=no_create)

        self.model_size = self.model_size + self.conv1.get_model_size() + self.conv2.get_model_size()
        self.flops = self.flops + self.conv1.get_flops(resolution=1) + self.conv2.get_flops(resolution=1) \
            + self.btn_channels + self.out_channels  # add relu flops
        self.activation_function = F.relu


    def forward(self, x):
        output = F.adaptive_avg_pool3d(x, output_size=(1,1,1))
        output = self.activation_function(self.conv1(output))
        output = self.conv2(output)
        output = F.sigmoid(output)
        output = output * x

        return output

    def get_model_size(self):
        return self.model_size

    def get_flops(self, resolution):
        return self.flops

    def get_layers(self):
        return 2

    def get_num_channels_list(self):
        return [self.btn_channels, self.out_channels]

    def get_log_zen_score(self, **kwarg):
        return None


class BaseSuperBlock3D(nn.Module):
    def __init__(self, structure_info, no_create=False, inner_class=None,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'BaseSuperBlock3D',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'kt': kernel_3d,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
            'inner_class': inner_class,
            'force_resproj_skip': force_resproj_skip (default=4),
        }
        :param NAS_mode:
        '''

        super().__init__()

        if 'class' in structure_info:
            assert structure_info['class'] == self.__class__.__name__

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
                force_resproj = False
            elif block_id % self.force_resproj_skip == 0:
                in_channels = self.out_channels
                out_channels = self.out_channels
                stride = 1
                # Note that: False for ZenDet, True for Madnas (linming)
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

            the_block = self.inner_class(structure_info=inner_structure_info,
                                         no_create=no_create,
                                         dropout_channel=self.dropout_channel,
                                         dropout_layer=self.dropout_layer)

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


    def get_model_size(self):
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


    def entropy_forward(self, x, skip_relu=True, skip_bn=True):
        output = x
        output_std_list = []
        for the_block in self.block_list:
            output, output_std_list_plain = the_block.entropy_forward(output, skip_relu=skip_relu, skip_bn=skip_bn)
            output_std_list += output_std_list_plain
        return output, output_std_list


    def get_num_channels_list(self):
        num_channels_list = []
        for block in self.block_list:
            num_channels_list += block.get_num_channels_list()

        return num_channels_list

    def get_stentr_forward(self, **kwarg):
        output_std_list_plain = []
        for block in self.block_list:
            output_std_list_plain += block.get_stentr_forward(**kwarg)
        return output_std_list_plain

    def get_log_zen_score(self, **kwarg):
        output_std_list_plain = []
        for block in self.block_list:
            output_std_list_plain += block.get_log_zen_score(**kwarg)
        return output_std_list_plain


    def sym_get_arch_encoding(self):
        return [self.bottleneck_channels, self.out_channels, self.num_inner_layers]


    def sym_set_arch_encoding(self, arch_encoding):
        self.bottleneck_channels, self.out_channels, self.num_inner_layers = arch_encoding
        self.block_list[0].in_channels = self.in_channels
        self.block_list[0].bottleneck_channels = self.bottleneck_channels
        self.block_list[0].out_channels = self.out_channels
        self.block_list[1].in_channels = self.out_channels
        self.block_list[1].bottleneck_channels = self.bottleneck_channels
        self.block_list[1].out_channels = self.out_channels


    def sym_get_model_size(self):
        sym_model_size = self.block_list[0].get_model_size() + self.block_list[1].get_model_size() * (self.num_inner_layers - 1)
        return sym_model_size


    def sym_get_flops(self, resolution):
        sym_flops = self.block_list[0].get_flops(resolution)
        resolution = resolution / self.block_list[0].stride
        sym_flops = sym_flops + self.block_list[1].get_flops(resolution) * (self.num_inner_layers - 1)
        return sym_flops


    def sym_get_effective_score(self):
        sym_L = self.num_inner_layers
        sym_avg_width = torch.sqrt(self.bottleneck_channels * self.out_channels)
        sym_es = sym_avg_width / sym_L
        return sym_es


__module_blocks__ = {
    'Conv3DKXBN': Conv3DKXBN,
    'Conv3DKXBNRELU': Conv3DKXBNRELU,
    'SE3D': SE3D,
    'BaseSuperBlock3D': BaseSuperBlock3D,
}
