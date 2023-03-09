# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import argparse
import ast
import copy
import os
import sys
import time

import numpy as np

str_format = '%s %d %d %d %d %d %d %d %d %d\n'
Conv_type_dict = {0: 'Regular', 1: 'Depthwise'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.in')
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--save-file', type=str, default='sample')
    args = parser.parse_args()
    return args


def read_config_info(file):
    config_info = {}
    if file is not None:
        with open(file, 'r') as fid:
            config_str_infos = [x.strip() for x in fid.readlines()]
        for config_str in config_str_infos:
            if 'channel_ratio' in config_str:
                config_info[config_str.split(' ')[0]] = [
                    float(x) for x in config_str.split(' ')[1:]
                ]
            else:
                if len(config_str.split(' ')[1:]) == 1:
                    config_info[config_str.split(' ')[0]] = int(
                        config_str.split(' ')[1])
                else:
                    config_info[config_str.split(' ')[0]] = [
                        int(x) for x in config_str.split(' ')[1:]
                    ]

    return config_info


def bench_single_t40(params_list):
    # params_list = In_H, In_W, In_C, nbitsA_in, Out_C, nbitsA_out, Kernel, Stride, nbitsW
    params_list_str = ' '.join([str(x) for x in params_list])
    cmd_t40 = './venus_eval_test_uclibc %s' % (params_list_str)
    try:
        cmd_return = os.popen(cmd_t40)
        results = cmd_return.readlines()
        latency = results[-1].split(' ')[2]
        latency = float(latency)
    except Exception:
        latency = -1
    return latency


def generateInputH(config_info):
    mMinH = config_info['min_feature_size']
    mMaxH = config_info['max_feature_size']
    mHPoints = config_info['number_feature_size']
    start_exp = np.log2(mMinH)
    end_exp = np.log2(mMaxH)
    step_exp = (end_exp - start_exp) / (mHPoints - 1)

    InputH_list = []
    InputH_list.append(mMinH)
    c = start_exp
    for i in range(mHPoints - 2):
        c += step_exp
        dH = int(np.exp2(c))
        # print(i, c, dH)
        if dH % 2 != 0:
            dH += 1
        InputH_list.append(dH)
    InputH_list.append(mMaxH)
    return InputH_list


def check_param_valid(mParam):
    # {Conv_type, Batch, In_C, In_H, In_W, Out_C, Kernel, Stride, ElmtFused} Latency
    if (mParam['In_C'] % 32 != 0 or mParam['Out_C'] % 32 != 0):
        return False
    if mParam['Conv_type'] == 'Regular':
        # if (mParam["In_C"] > 2048 and mParam["Out_C"] > 2048 and mParam["In_H"] >= 20): return False
        # if (mParam["In_C"] > 1024 and mParam["Out_C"] > 1024 and mParam["Kernel"] >= 5): return False
        if (mParam['In_C'] != 3 and mParam['In_C'] < 8):
            return False
        if (mParam['In_C'] > 4096 or mParam['Out_C'] > 4096):
            return False
        if (mParam['In_C'] == 3 and mParam['Out_C'] > 256):
            return False

    if (mParam['Stride'] == 2 and mParam['In_H'] % 2 != 0):
        return False
    if (mParam['Conv_type'] == 'Depthwise'
            and mParam['In_C'] != mParam['Out_C']):
        return False
    if (mParam['Conv_type'] == 'Depthwise' and mParam['ElmtFused'] == 1):
        return False
    if (mParam['Conv_type'] == 'Depthwise' and mParam['Kernel'] > 7):
        return False

    input_size = mParam['Batch'] * mParam['In_H'] * mParam['In_W'] * mParam[
        'In_C']
    output_size = mParam['Batch'] * mParam['In_H'] * mParam['In_W'] * mParam[
        'Out_C']
    if (mParam['Stride'] == 2):
        output_size /= 4

    tensor_size_thres = 128 * 240 * 240 * 128
    if (input_size > tensor_size_thres or output_size > tensor_size_thres):
        return False

    flops = 1.0 * output_size * mParam['In_C'] * mParam['Kernel'] * mParam[
        'Kernel']
    flops_thres = 78.0 * 78 * 3072 * 1 * 1 * 3072 * 4
    if mParam['Conv_type'] == 'Depthwise':
        flops /= mParam['In_C']
    if (flops > flops_thres):
        return False

    return True


def check_list(elmt_list):
    if isinstance(elmt_list, list):
        return elmt_list
    elif isinstance(elmt_list, int):
        return [elmt_list]
    else:
        raise ValueError('elmt_list must be a int or a list, not %s: %s' %
                         (type(elmt_list), elmt_list))


def generate_mParam_list(InputH_list, config_info):
    # {Conv_type, Batch, In_C, In_H, In_W, Out_C, Kernel, Stride, ElmtFused} Latency
    ElmtFused_list = check_list(config_info['elmt_fused'])[::-1]
    Stride_list = check_list(config_info['stride'])[::]
    Kernel_list = check_list(config_info['filter_size'])[::-1]
    Out_C_list = check_list(config_info['output_channel'])[::-1]
    In_C_ratio_list = check_list(config_info['channel_ratio'])
    print(In_C_ratio_list)
    Batch_list = check_list(config_info['batch'])[::-1]
    Conv_type_list = check_list(config_info['type'])[::-1]

    mParam_list = []
    for ElmtFused in ElmtFused_list:
        for Batch in Batch_list:
            for Kernel in Kernel_list:
                for In_H in InputH_list[::-1]:
                    for Conv_type in Conv_type_list:
                        for Out_C in Out_C_list:
                            for Stride in Stride_list:
                                for idx in range(len(In_C_ratio_list) + 1):
                                    if idx == 0:
                                        In_C = 3
                                    else:
                                        In_C = int(In_C_ratio_list[idx - 1]
                                                   * Out_C)
                                    mParam = {
                                        'Conv_type': Conv_type_dict[Conv_type],
                                        'Batch': Batch,
                                        'In_C': In_C,
                                        'In_H': In_H,
                                        'In_W': In_H,
                                        'Out_C': Out_C,
                                        'Kernel': Kernel,
                                        'Stride': Stride,
                                        'ElmtFused': ElmtFused
                                    }
                                    # print("\n", mParam)
                                    # time.sleep(1)
                                    if check_param_valid(mParam):
                                        mParam_list.append(mParam)
    return mParam_list


def sample_for_config(config_info, save_file, nbits):
    with open(save_file, 'w+') as fw:
        InputH_list = generateInputH(config_info)
        print('==> the InputH is: ', InputH_list)
        mParam_list = generate_mParam_list(InputH_list, config_info)
        print('==> the valid sample num is: %d' % (len(mParam_list)))

        for mParam in mParam_list:
            # {Conv_type, Batch, In_C, In_H, In_W, Out_C, Kernel, Stride, ElmtFused} Latency
            fw.write(
                str_format %
                (mParam['Conv_type'], mParam['Batch'], mParam['In_C'],
                 mParam['In_H'], mParam['In_W'], mParam['Out_C'],
                 mParam['Kernel'], mParam['Stride'], mParam['ElmtFused'], 0))


def main():
    args = parse_args()
    config_info = read_config_info(args.config)
    print(config_info)
    args.save_file = '%s.int%d.txt' % (args.save_file, args.nbits)
    sample_for_config(config_info, args.save_file, args.nbits)


if __name__ == '__main__':
    main()
