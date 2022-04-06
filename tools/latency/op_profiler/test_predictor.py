# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

###
import os
import sys
import pdb
import torch
import argparse
import numpy as np
import logging 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs import load_py_module_from_path
from nas.models import MasterNet
import predictor as opp


def predict_the_latency(save_dir, plainnet_struct, resolution, \
                device_name="V100", date_type="FP16", bs_list=[1,32,64], save_onnx=False):

    # pdb.set_trace()
    fw = open("%s/conv_test.out"%(device_name), "w")
    the_model = MasterNet(num_classes=1000, structure_txt=plainnet_struct,
                        no_create=False, out_indices=(4,))
    print("the model is %s\n"%(str(the_model)))

    net_params = the_model.get_params_for_trt(resolution)
    # print(net_params)
    # remove other params, only conv and convDW
    net_params_conv = []
    for idx, net_param in enumerate(net_params):
        if net_param[0] in ["Regular", "Depthwise"]:
                net_params_conv.append(net_param)
                # print("idx %d: %s"%(idx, net_param))
    times = [0]*len(net_params_conv)
    for net_param_conv in net_params_conv:
        fw.write("%s %s\n"%(net_param_conv, 0))

    Predictor_opp = opp.OpProfiler(device_name=device_name, date_type=date_type)
    the_latency_list = []
    for batchsize in bs_list:
        _, the_latency = Predictor_opp(net_params_conv, times, batchsize)
        the_latency = the_latency/batchsize
        the_latency_list.append(the_latency_list)
        print("batchsize: %d, the TensorRT predict Latency is %.4f ms"%(batchsize, the_latency))

    fw.close()

    if save_onnx:
        for batchsize in bs_list:
            x = torch.randn(batchsize, 3, resolution, resolution, requires_grad=False)
            out_name = os.path.join(save_dir, "%s/R50_bs%d.onnx"%(device_name, batchsize))
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            torch.onnx.export(the_model, x, out_name, input_names = ['input']) 

    return the_latency_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_dir', type=str, default="R50")
    parser.add_argument('-p', '--plainnet_struct', type=str, default="R50.txt")
    parser.add_argument('-d', '--device_name', type=str, default="T40")
    parser.add_argument('-dt', '--date_type', type=str, default="INT8")
    parser.add_argument('-r', '--resolution', type=int, default=224)
    parser.add_argument('-bs', '--bs_list', nargs='+', type=int, default=[1])
    parser.add_argument('--onnx', action='store_true', help="wether to save onnx")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    log_level = logging.DEBUG
    args = parse_args()
    save_dir = args.save_dir
    plainnet_struct = args.plainnet_struct
    device_name = args.device_name
    date_type = args.date_type
    resolution = args.resolution
    bs_list = args.bs_list
    the_latency_list = predict_the_latency(save_dir, plainnet_struct, resolution, \
        device_name=device_name, date_type=date_type, bs_list=bs_list, save_onnx=args.onnx)