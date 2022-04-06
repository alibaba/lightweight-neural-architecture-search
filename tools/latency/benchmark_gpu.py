# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys, argparse
import numpy as np
import torch, logging, time


def __get_latency__(model, batch_size, resolution, channel, gpu, benchmark_repeat_times, fp16):
    device = torch.device('cuda:{}'.format(gpu))
    torch.backends.cudnn.benchmark = True

    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32
    if type(resolution)==list and len(resolution)==2:
        the_image = torch.randn(batch_size, channel, resolution[0], resolution[1], dtype=dtype,
                            device=device)
    else:
        the_image = torch.randn(batch_size, channel, resolution, resolution, dtype=dtype,
                            device=device)
    model.eval()
    warmup_T = 3
    with torch.no_grad():
        for i in range(warmup_T):
            the_output = model(the_image)
        start_timer = time.time()
        for repeat_count in range(benchmark_repeat_times):
            the_output = model(the_image)

    end_timer = time.time()
    the_latency = (end_timer - start_timer) / float(benchmark_repeat_times) / batch_size
    return the_latency


class GetRobustLatencyMeanStd():
    def __init__(self, batch_size, resolution, gpu, channel=3, fp16=False, benchmark_repeat_times=30):
        self.batch_size = batch_size
        self.resolution = resolution
        self.gpu = gpu
        self.channel = channel
        self.benchmark_repeat_times = benchmark_repeat_times
        self.fp16 = fp16


    def forward(self, model):

        robust_repeat_times = 10
        latency_list = []
        model = model.cuda(self.gpu)
        for repeat_count in range(robust_repeat_times):
            try:
                the_latency = __get_latency__(model, self.batch_size, self.resolution, 
                            self.channel, self.gpu, self.benchmark_repeat_times, self.fp16)
            except Exception as e:
                print(e)
                the_latency = np.inf

            latency_list.append(the_latency)

        pass  # end for
        latency_list.sort()
        avg_latency = np.mean(latency_list[2:8])
        std_latency = np.std(latency_list[2:8])
        return avg_latency, std_latency


def main():
    pass


if __name__ == "__main__":
    main()