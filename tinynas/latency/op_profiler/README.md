***
![framework](op_profiler.jpg)
***
## Introduction

<!-- English | [简体中文](README_zh-CN.md) -->

The latency predictor based on Conv operations profiler, supporting Nvidia GPUs.

Thanks to the sequential structure of the neural network, we can approximate the latency of the model by summing up the latency of each layer.

First build a operations latency library containing the latency of sampling Conv operations under a large [configurations](python/config.in).

Then use the `predictor.py` to predict the summary latency of each layer in the target network structure. The operation which is not in the library is obtained by Scipy Interpolations between other operations.


***
<details open>
<summary>Major features</summary>

* **Sampling process with python example**

    `python`: the code how to generate the operations latency library on target device. [How to use](python/README.md).

* **Nvidia GPUs TRT latency**

    `V100`: the sampling operations latency library for Nvidia GPU V100 based TRT with precision of FP32/FP16/INT8

    Other GPUs like P100/T4, could use the sampling code to generate the library like V100.

</details>

***

## Format for each element in the library

```python
{Conv_type, Batch, In_C, In_H, In_W, Out_C, Kernel, Stride, ElmtFused} Latency
```
* Conv_type: 'Regular' or 'Depthwise' convolution
* Batch: batch size, typical = 1, 32, 64, 128
* In_C: input_channels; Out_C: output_channels; In_C=Out_C*Ratio
* In_H: input_height; In_W: input_width
* Kernel: kernel size, typical = 1, 3, 5, 7
* Stride: stride value for convolution, typical = 1, 2
* ElmtFused: whether the elementwise sum operation of the relink structure is included
* Latency: the profiling time of each element


## Format for each element in the predictor
[("Regular", self.stride, elmtfused, self.kernel_size, 1, self.in_channels, input_resolution, self.out_channels)]

```python
[Conv_type, Stride, ElmtFused, Kernel, Batch, In_C, In_H, Out_C]
```
* Conv_type: 'Regular' or 'Depthwise'
* Stride: stride value for convolution, typical = 1, 2
* ElmtFused: whether the elementwise sum operation of the relink structure is included
* Kernel: kernel size, typical = 1, 3, 5, 7
* Batch: 1, specific batchsize in a hyper parameter for the predictor
* In_C: input_channels; Out_C: output_channels
* In_H: the width the feature map is equal to the height
