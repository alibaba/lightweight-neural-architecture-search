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

# {Conv_type, Batch, In_C, In_H, In_W, Out_C, Kernel, Stride, ElmtFused} Latency
str_format = '%s %d %d %d %d %d %d %d %d %d\n'
Conv_type_dict = {0: 'Regular', 1: 'Depthwise'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='log')
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--save_file', type=str, default='../conv_data.out')
    args = parser.parse_args()
    return args


def main():
    import pdb
    args = parse_args()
    lines = open('%s.int%d.txt' %
                 (args.log_file, args.nbits)).read().splitlines()
    iter_flag = False
    lib_lines = []
    for line in lines:
        if 'iter-' in line:
            if iter_flag:
                exit()
            params = line.split(': ')[1].split(' ')
            params_line = '{%s}' % (','.join(params[:-1]))
            print(params_line)
            iter_flag = True
        if iter_flag and 'run time_ms: ' in line:
            iter_flag = False
            # "run time_ms: 4123.077000 ms"
            latency = float(line.split(' ')[2])
            lib_lines.append('%s %f' % (params_line, latency))
            print(latency)
    with open(args.save_file + '.int%d' % (args.nbits), 'w') as fw:
        fw.writelines('\n'.join(lib_lines))


if __name__ == '__main__':
    main()
