# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from .builder import LATENCIES


def __get_latency__(model, batch_size, image_size, channel, gpu,
                    benchmark_repeat_times, fp16):
    device = torch.device('cuda:{}'.format(gpu))
    torch.backends.cudnn.benchmark = True

    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32
    if type(image_size) == list and len(image_size) == 2:
        the_image = torch.randn(
            batch_size,
            channel,
            image_size[0],
            image_size[1],
            dtype=dtype,
            device=device)
    else:
        the_image = torch.randn(
            batch_size,
            channel,
            image_size,
            image_size,
            dtype=dtype,
            device=device)
    model.eval()
    warmup_T = 3
    with torch.no_grad():
        for i in range(warmup_T):
            _ = model(the_image)
        start_timer = time.time()
        for repeat_count in range(benchmark_repeat_times):
            _ = model(the_image)

    end_timer = time.time()
    the_latency = (end_timer
                   - start_timer) / float(benchmark_repeat_times) / batch_size
    return the_latency

@LATENCIES.register_module(module_name = 'RobustGpuPredictor')
class RobustGpuPredictor():

    def __init__(self,
                 batch_size,
                 image_size,
                 gpu ,
                 channel=3,
                 fp16=False,
                 repeat_times=30,
                 **kwargs):
        assert isinstance(gpu, int), 'Must set enable_gpu=True in cfg' 
        self.batch_size = batch_size
        self.image_size = image_size
        self.gpu = gpu
        self.channel = channel
        self.repeat_times = repeat_times
        self.fp16 = fp16
        assert isinstance(gpu, int) 

    def __call__(self, model):

        robust_repeat_times = 10
        latency_list = []
        model = model.cuda(self.gpu)
        for repeat_count in range(robust_repeat_times):
            try:
                the_latency = __get_latency__(model, self.batch_size,
                                              self.image_size, self.channel,
                                              self.gpu,
                                              self.repeat_times,
                                              self.fp16)
            except Exception as e:
                print(e)
                the_latency = np.inf

            latency_list.append(the_latency)

        pass  # end for
        latency_list.sort()
        avg_latency = np.mean(latency_list[2:8])
        std_latency = np.std(latency_list[2:8])
        return avg_latency #, std_latency


def main():
    pass


if __name__ == '__main__':
    main()
