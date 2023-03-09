# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import argparse
import logging
###
import os

import predictor as opp
import torch
from nas.models import MasterNet


def predict_the_latency(save_dir,
                        plainnet_struct,
                        resolution,
                        device_name='V100',
                        date_type='FP16',
                        bs_list=[1, 32, 64],
                        save_onnx=False):

    fw = open('%s/conv_test.out' % (device_name), 'w')
    the_model = MasterNet(
        num_classes=1000,
        structure_txt=plainnet_struct,
        no_create=False,
        out_indices=(4, ))
    print('the model is %s\n' % (str(the_model)))

    net_params = the_model.get_params_for_trt(resolution)
    # print(net_params)
    # remove other params, only conv and convDW
    net_params_conv = []
    for idx, net_param in enumerate(net_params):
        if net_param[0] in ['Regular', 'Depthwise']:
            net_params_conv.append(net_param)
            # print("idx %d: %s"%(idx, net_param))
    times = [0] * len(net_params_conv)
    for net_param_conv in net_params_conv:
        fw.write('%s %s\n' % (net_param_conv, 0))

    Predictor_opp = opp.OpProfiler(
        device_name=device_name, date_type=date_type)
    the_latency_list = []
    for batchsize in bs_list:
        _, the_latency = Predictor_opp(net_params_conv, times, batchsize)
        the_latency = the_latency / batchsize
        the_latency_list.append(the_latency_list)
        print('batchsize: %d, the TensorRT predict Latency is %.4f ms' %
              (batchsize, the_latency))

    fw.close()

    if save_onnx:
        for batchsize in bs_list:
            x = torch.randn(
                batchsize, 3, resolution, resolution, requires_grad=False)
            out_name = os.path.join(
                save_dir, '%s/R50_bs%d.onnx' % (device_name, batchsize))
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            torch.onnx.export(the_model, x, out_name, input_names=['input'])

    return the_latency_list
