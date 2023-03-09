# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

__all__ = ['QConv2d']
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import BatchNorm2d, Conv2d, Linear
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Module):

    def __init__(self, bitwidth, positive=False, quant_search=False):
        super(LsqQuan, self).__init__()
        if positive:
            self.lower_bound = 0
            self.upper_bound = 2**bitwidth - 1
        else:
            self.lower_bound = -2**(bitwidth - 1)
            if quant_search:
                self.upper_bound = 2**(bitwidth - 1)
            else:
                # actual need to minus 1
                self.upper_bound = 2**(bitwidth - 1) - 1

        self.s = Parameter(torch.ones(1))

    def init_from(self, x):
        self.s = Parameter(
            x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)
            * 2 / (self.upper_bound**0.5))

    def forward(self, x):
        s_grad_scale = 1.0 / ((self.upper_bound * x.numel())**0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.lower_bound, self.upper_bound)
        x_q = round_pass(x)
        x = x_q * s_scale
        return x, s_scale.reshape((s_scale.shape[0])), x_q


class QuanActivation(Function):

    @staticmethod
    def forward(ctx, arr, bitwidth, positive=False):
        if positive:
            upper_bound = 2**(bitwidth) - 1
            lower_bound = 0
        else:
            upper_bound = 2**(bitwidth - 1) - 1
            lower_bound = -2**(bitwidth - 1)

        abs_arr = torch.abs(arr)
        epsilon = 1e-10
        alpha = torch.max(abs_arr) / upper_bound
        val_q = arr / (alpha + epsilon)
        arr_q = torch.round(torch.clamp(val_q, lower_bound, upper_bound))
        arr_f = arr_q * alpha
        return arr_f, alpha

    @staticmethod
    def backward(ctx, grad_output, useless_1):
        grad_arr = None
        if ctx.needs_input_grad[0]:
            grad_arr = grad_output
        return grad_arr, None


class QuanWeight(Function):

    @staticmethod
    def forward(ctx,
                arr,
                bitwidth,
                positive=False,
                maxiter=20,
                maxdiff=0.0001,
                bUseMultiAlpha=True):
        if positive:
            upper_bound = 2**(bitwidth) - 1
            lower_bound = 0
        else:
            upper_bound = 2**(bitwidth - 1) - 1
            lower_bound = -2**(bitwidth - 1)

        if (bitwidth == 1):
            upper_bound = 1
            lower_bound = 0
        ker_num = 1
        for i in arr.shape[1:]:
            ker_num = ker_num * i
        arr_f = arr.reshape((arr.shape[0], ker_num))
        abs_arr = torch.abs(arr_f)
        if bUseMultiAlpha:
            axisindex = 1
            sumCount = ker_num
        else:
            axisindex = None
            sumCount = arr.size
        if (quan_type > 3):
            alpha = torch.max(abs_arr, dim=axisindex)[0] / (upper_bound * 1.25)
        else:
            alpha = torch.sum(abs_arr, dim=axisindex) / float(sumCount)
        # ADMM
        PreSum = torch.sum(torch.abs(alpha))
        n = 0
        epsilon = 1e-10
        DiffRate = 1
        while ((n < maxiter) and (DiffRate >= maxdiff)):
            val_q = arr_f / (alpha.reshape((alpha.shape[0], 1)) + epsilon)
            # update Quantizer
            arr_q = torch.round(torch.clamp(val_q, lower_bound, upper_bound))
            # update Alpha
            w1 = torch.sum(torch.mul(arr_q, arr_f), dim=axisindex)
            w2 = torch.sum(torch.mul(arr_q, arr_q), dim=axisindex)
            alpha = w1 / (w2 + epsilon)
            #
            CurSum = torch.sum(torch.abs(alpha))
            DiffRate = torch.abs(CurSum - PreSum) / PreSum
            PreSum = CurSum
            n = n + 1
        arr_f = arr_q * alpha.reshape((alpha.shape[0], 1))
        return arr_f.reshape(arr.shape), alpha, arr_q.reshape(arr.shape)

    @staticmethod
    def backward(ctx, grad_output, useless_1, useless_2):
        grad_arr = None
        if ctx.needs_input_grad[0]:
            grad_arr = grad_output
        return grad_arr, None, None, None


class QConv2d(Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 nbitsA=8,
                 nbitsW=8,
                 quan_type='lsq',
                 positive=False,
                 **kwargs):
        super(QConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        assert quan_type in ['lsq', 'admm']

        if 'quant_search' in kwargs:
            self.quant_search = kwargs['quant_search']
        else:
            self.quant_search = False

        self.nbitsA = nbitsA
        self.nbitsW = nbitsW
        self.quan_type = quan_type
        self.positive = positive
        self.weightq = Parameter(
            torch.zeros(self.weight.shape), requires_grad=False)
        self.alpha = Parameter(torch.ones((out_channels)), requires_grad=False)
        self.scalar = Parameter(torch.ones(1), requires_grad=False)

        if quan_type == 'lsq':
            self.act_quan = LsqQuan(
                self.nbitsA, positive, quant_search=self.quant_search)
            self.weight_quan = LsqQuan(
                self.nbitsW, positive, quant_search=self.quant_search)
            if not self.quant_search:
                self.weight_quan.init_from(self.weight)

    def forward(self, input):
        if self.quan_type == 'lsq':
            qact, self.scalar.data, _ = self.act_quan(input)
            w_f, self.alpha.data, self.weightq.data = self.weight_quan(
                self.weight)
        else:
            qact, self.scalar.data = QuanActivation.apply(
                input, self.nbitsA, self.positive)
            w_f, self.alpha.data, self.weightq.data = QuanWeight.apply(
                self.weight, self.nbitsW, self.positive)
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(
                    qact,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode), w_f, self.bias, self.stride,
                _pair(0), self.dilation, self.groups)
        else:
            return F.conv2d(qact, w_f, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)

    def extra_repr(self):
        return super(QConv2d, self).extra_repr() + ', nbitsA={}, nbitsW={}, quan_type={}, positive={}'\
            .format(self.nbitsA, self.nbitsW, self.quan_type, self.positive)


class QLinear(Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 nbits=4,
                 quan_type='lsq',
                 positive=False):
        super(QLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias)

        self.nbits = nbits
        self.quan_type = quan_type
        self.positive = positive
        self.weightq = Parameter(
            torch.zeros(self.weight.shape), requires_grad=False)
        self.alpha = Parameter(torch.ones((out_features)), requires_grad=False)
        self.scalar = Parameter(torch.ones(1), requires_grad=False)

        if quan_type == 'lsq':
            self.act_quan = LsqQuan(nbits, positive)
            self.weight_quan = LsqQuan(nbits, positive)
            self.weight_quan.init_from(self.weight)

    def forward(self, input):
        if self.quan_type == 'lsq':
            qact, self.scalar.data, _ = self.act_quan(input)
            w_f, self.alpha.data, self.weightq.data = self.weight_quan(
                self.weight)
        else:
            qact, self.scalar.data = QuanActivation.apply(
                input, self.nbits, self.positive)
            w_f, self.alpha.data, self.weightq.data = QuanWeight.apply(
                self.weight, self.nbits, self.positive)

        return F.linear(qact, w_f, self.bias)

    def extra_repr(self):
        return super(QLinear, self).extra_repr() + ', nbits={}, quan_type={}, positive={}' \
            .format(self.nbits, self.quan_type, self.positive)


class QBatchNorm2d(nn.Module):

    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 nbits=8,
                 quan_type='lsq',
                 positive=False):
        super().__init__()
        assert quan_type in ['lsq', 'admm']

        self.nbits = nbits
        self.quan_type = quan_type
        self.positive = positive

        self.scalar = Parameter(torch.ones((1)), requires_grad=False)
        self.bn = BatchNorm2d(
            num_features,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        if quan_type == 'lsq':
            self.act_quan = LsqQuan(nbits, positive)

    def forward(self, input):
        out = self.bn(input)
        if self.quan_type == 'lsq':
            qact, self.scalar.data = self.act_quan(out)
        else:
            qact, self.scalar.data = QuanActivation.apply(
                out, self.nbits, self.positive)
        return qact

    def extra_repr(self):
        return super(QBatchNorm2d, self).extra_repr() + ', nbits={}, quan_type={}, positive={}' \
            .format(self.nbits, self.quan_type, self.positive)


class QConv2dv2(Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 nbits=8,
                 quan_type='lsq',
                 positive=False):
        super(QConv2dv2,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        assert quan_type in ['lsq', 'admm']

        self.nbits = nbits
        self.quan_type = quan_type
        self.positive = positive
        self.weightq = Parameter(
            torch.zeros(self.weight.shape), requires_grad=False)
        self.alpha = Parameter(torch.ones((out_channels)), requires_grad=False)

        if quan_type == 'lsq':
            self.weight_quan = LsqQuan(nbits, positive)
            self.weight_quan.init_from(self.weight)

    def forward(self, input):
        if self.quan_type == 'lsq':
            w_f, self.alpha.data, self.weightq.data = self.weight_quan(
                self.weight)
        else:
            w_f, self.alpha.data, self.weightq.data = QuanWeight.apply(
                self.weight, self.nbits, self.positive)

        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode), w_f, self.bias, self.stride,
                _pair(0), self.dilation, self.groups)
        else:
            return F.conv2d(input, w_f, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)

    def extra_repr(self):
        return super(QConv2dv2, self).extra_repr() + ', nbits={}, quan_type={}, positive={}' \
            .format(self.nbits, self.quan_type, self.positive)


class QAct(nn.Module):

    def __init__(self, nbits=8, quan_type='lsq', positive=False):
        super(QAct, self).__init__()

        assert quan_type in ['lsq', 'admm']

        self.nbits = nbits
        self.quan_type = quan_type
        self.positive = positive
        self.scalar = Parameter(torch.ones(1), requires_grad=False)

        if quan_type == 'lsq':
            self.act_quan = LsqQuan(nbits, positive)

    def forward(self, input):
        if self.quan_type == 'lsq':
            qact, self.scalar.data, _ = self.act_quan(input)
        else:
            qact, self.scalar.data = QuanActivation.apply(
                input, self.nbits, self.positive)
        return qact

    def extra_repr(self):
        return super(QAct, self).extra_repr() + ', nbits={}, quan_type={}, positive={}' \
            .format(self.nbits, self.quan_type, self.positive)
