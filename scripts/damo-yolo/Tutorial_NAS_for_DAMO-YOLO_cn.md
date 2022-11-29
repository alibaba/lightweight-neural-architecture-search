- 目标：基于LightNAS最新版本，添加新的搜索空间、结构模板等模块，搜索出用于DAMO-YOLO的基础结构
- 本文以k1kx基础结构为例进行介绍，其他结构（如kxkx）的搜索与开发流程类似
- 本文中涉及到的代码修改已经准备就绪，可以直接按照[环境准备](#环境准备)和[DAMO-YOLO backbone结构搜索](#DAMO-YOLO-backbone结构搜索)章节使用。若您想自定义自己的搜索过程，可以参考本文的[进阶教程：详细代码修改](#进阶教程详细代码修改)章节进行自定义开发

# 目录
[TOC]

# 环境准备
## 基础环境需求

- LInux
- CUDA 10.2+
- conda
## 相关依赖库安装

1. 获取LightNAS代码库
```shell
git clone https://github.com/alibaba/lightweight-neural-architecture-search.git
```

2. 创建一个新的conda环境并激活它
```shell
conda create -n light-nas python=3.6 -y
conda activate light-nas
```

3. 用conda安装openmpi以及mpi4py（此方法需要cuda 10.2及以上，安装完成后可从requirements.txt中删除mpi4py的pip安装依赖；也可以按照代码库的README先安装openmpi，再通过pip安装mpi4py）
```shell
conda install openmpi mpi4py
```

4. 通过conda安装torch和torchvision，或者参考[官方指令](https://pytorch.org/get-started/locally/)
```shell
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
```

5. 安装所需的其它依赖库
```shell
cd lightweight-neural-architecture-search
pip install -r requirements.txt
```

   - 注：由于openmpi和mpi4py已经从conda安装，再从pip安装mpi4py可能会失败，因此可以在requirements.txt中删除mpi4py的依赖条目

# DAMO-YOLO backbone结构搜索

- 搜索过程很简单，只需要进入到对应的DAMO-YOLO搜索脚本目录，然后运行脚本即可
```shell
cd scripts/damo-yolo
sh example_k1kx_small.sh
```
- 不同scale的模型（tiny，small，medium）可用不同的脚本搜索
  - 注：tiny模型的stage宽度限制范围需要更小，最大分别为`[96, 192, 384]`，在搜索时可参考[k1kx搜索空间添加过程](#k1kx搜索空间添加过程)小节的4.iv.a部分对相应代码进行修改

- 搜索结束后，成果物会保存于工作空间中。其中，分数最高的结构会保存于best_structure.txt文件中，而包含best_structure及其相关信息的内容会保存于nas_info.txt
<details>
<summary>搜索结果nas_info.txt示例</summary>

```
{ 'acc': 1399.2675807481487,
  'flops': 7351091200.0,
  'latency': 0.00011991791411210322,
  'layers': 25,
  'params': 5831696.0,
  'score': 1399.2675807481487,
  'stages': 4,
  'structure': [ {'class': 'ConvKXBNRELU', 'in': 12, 'k': 3, 'out': 32, 's': 1},
                 { 'L': 1,
                   'btn': 24,
                   'class': 'SuperResConvK1KX',
                   'in': 32,
                   'inner_class': 'ResConvK1KX',
                   'k': 3,
                   'out': 96,
                   's': 2},
                 { 'L': 2,
                   'btn': 96,
                   'class': 'SuperResConvK1KX',
                   'in': 96,
                   'inner_class': 'ResConvK1KX',
                   'k': 3,
                   'out': 96,
                   's': 2},
                 { 'L': 3,
                   'btn': 128,
                   'class': 'SuperResConvK1KX',
                   'in': 96,
                   'inner_class': 'ResConvK1KX',
                   'k': 3,
                   'out': 256,
                   's': 2},
                 { 'L': 3,
                   'btn': 144,
                   'class': 'SuperResConvK1KX',
                   'in': 256,
                   'inner_class': 'ResConvK1KX',
                   'k': 3,
                   'out': 256,
                   's': 1},
                 { 'L': 3,
                   'btn': 224,
                   'class': 'SuperResConvK1KX',
                   'in': 256,
                   'inner_class': 'ResConvK1KX',
                   'k': 3,
                   'out': 512,
                   's': 2}]}
```
</details>

- 这个搜索到的结构可以经过[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) 官方推荐的wrapper（基于搜索结果模型添加Focus、SPP等模块）包装得到最终的backbone
  - backbone wrapper位置：[link](https://github.com/tinyvision/DAMO-YOLO/tree/master/damo/base_models/backbones)
    - [tinynas_res.py](https://github.com/tinyvision/DAMO-YOLO/blob/master/damo/base_models/backbones/tinynas_res.py) ：只带有Focus、SPP等模块的wrapper版本，用于T和S模型
    - [tinynas_csp.py](https://github.com/tinyvision/DAMO-YOLO/blob/master/damo/base_models/backbones/tinynas_csp.py) ：额外带有CSP模块的wrapper版本，用于M模型
    - 已搜索好的不同scale的结构文件可见[此处](https://github.com/tinyvision/DAMO-YOLO/tree/master/damo/base_models/backbones/nas_backbones)
  - 性能验证：将其替换yolox-s的backbone后性能如下：
    - 可以看到，根据本教程搜索到的backbone，经过wrapper包装后，最终性能和速度均可优于原yolox-s模型

| 模型 | GFLOPs | params (M) | mAP | trt latency on V100, bs=1, fp32 (ms) |
| --- | --- | --- | --- | --- |
| yolox-s | 26.93 | 8.96 | 40.5 | 4.69 |
| nas-for-damo-yolo | 32.30 | 11.68 | **41.1** | **4.57** |


# 进阶教程：详细代码修改

## 基础知识

- LightNAS主要逻辑部分的代码都在nas目录下，大致可分为6个部分：
   - evolutions：进化算法相关代码，与具体的网络结构无关
   - models：网络模板相关代码，规定了该框架支持哪些结构的网络搜索
   - scores：打分相关代码，提供了给每个网络计算其性能分数的方法，目前包括entropy和madnas两种
   - spaces：搜索空间相关代码，每种搜索空间在每次迭代时进行变化的方法和范围会有所不同
   - builder.py：构建整个nas框架的代码
   - search.py：网络搜索的主流程代码

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665579630065-5540fadb-be89-434d-a1d7-ad6023ab696c.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=142&id=udc40dbff&margin=%5Bobject%20Object%5D&name=image.png&originHeight=161&originWidth=194&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8300&status=done&style=none&taskId=ufae6dc6c-17c0-4b7d-a4b5-c9b9550d7fb&title=&width=171)

- 当前搜索空间包括3种：（见nas/spaces）
   - mobilenet-like：k1dwk1，bottleneck中间为depthwise卷积
   - resnet-like：k1kxk1，bottleneck中间为kernel size为x的常规卷积
   - 带量化的mobilenet-like：quan_k1dwk1，在mobilenet-like基础上考虑了混合精度量化

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665578853811-9ff818a4-c900-4aea-b6e5-548a86c52076.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=148&id=u47f710d3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=189&originWidth=259&originalType=binary&ratio=1&rotation=0&showTitle=false&size=12475&status=done&style=none&taskId=u9f97a2ed-88a8-4010-9440-7938187977c&title=&width=202.5)

- 搜索空间对应的结构模板共2种，每种又包括带量化和不带量化2个版本：（见nas/models/blocks）
   - SuperResConvK1KXK1：resnet-like的结构模板
   - SuperResK1DWK1：mobilenet-like的结构模板
   - 每种结构中以结构信息字符串中每个block是否带有nbitsA和nbitsW字段来判断其是否考虑混合精度量化
   - 结构信息字符串会由MasterNet解析以得到

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665578886026-733bfe2a-107d-459c-97a5-4ceaf28a09c6.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=194&id=u76e7c9e9&margin=%5Bobject%20Object%5D&name=image.png&originHeight=248&originWidth=280&originalType=binary&ratio=1&rotation=0&showTitle=false&size=17272&status=done&style=none&taskId=ud5cb7010-5ed2-42e4-86d4-be535a0a729&title=&width=219)

- 根据针对的任务类型与目标不同，当前代码共提供了3中类型的搜索脚本：（见scripts）
   - classification：针对分类任务
      - 特点为在搜索的backbone最后，MasterNet会根据类别数目自动添加进行分类的线性层
      - 提供了两种结构模板的示例代码
   - detection：针对检测任务
      - 特点为仅输出backbone部分，可指定需要输出特征的block编号，使用时需自行添加之后的neck和head等部分
      - 只提供了resnet-like结构的示例代码
   - quant：考虑混合精度量化目标
      - 特点为初始结构中添加了nbitsA和nbitsW两个字段，因此搜索时会综合考虑各个block的量化精度
      - 目前暂时只提供了mobilenet-like结构的示例代码

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665578917658-9e1b0a6b-5740-4a63-a3a6-36b77a9415dc.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=313&id=u87714381&margin=%5Bobject%20Object%5D&name=image.png&originHeight=365&originWidth=269&originalType=binary&ratio=1&rotation=0&showTitle=false&size=27347&status=done&style=none&taskId=u0c75cb31-38bd-49ac-a0aa-586b07ceb96&title=&width=230.5)

- 进行搜索时，可对一系列参数进行配置（完整参数见configs/config_nas.py）
   - 具体调节方式可参考各个搜索脚本
   - 包括网络初始结构，FLOPs、model_size等各种搜索限制，搜索空间，进化算法配置等等

> 综上，我们要针对DAMO-YOLO搜索合适的k1kx网络结构，大致需要进行以下几项工作：
> 
> - 为新的k1kx结构模板添加block代码 --> nas/models/blocks
> - 为新的k1kx结构添加搜索空间代码 --> nas/spaces
> - 为新的k1kx结构添加搜索脚本代码 --> scripts/damo-yolo
> - 其他细节代码调整


## 结构模板添加

### k1kx结构

- k1kx结构类似于CSPDarknet中的bottleneck部分，它和k1kxk1结构非常类似
- 不同于k1kxk1的前后两个k1分别用于降维和升维而中间的kx仅用于特征变换，k1kx结构中k1用于降维，而kx进行特征变换的同时也进行升维
- 因此我们可以基于已有的k1kxk1的结构模板（即nas/models/blocks/SuperResConvK1KXK1.py）进行改造

### 添加过程

1. 在nas/models/blocks目录下，复制SuperResConvK1KXK1.py为新的模板SuperResConvK1KX.py
2. 将基础模块类名由ResConvK1KXK1修改为ResConvK1KX

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665631433761-4a66c630-81e9-4b77-adbd-036e3820b37f.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=158&id=ud144dfcb&margin=%5Bobject%20Object%5D&name=image.png&originHeight=216&originWidth=698&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31367&status=done&style=none&taskId=u642a97cb-a10e-4df5-859d-ae2a9689d10&title=&width=511)

3. __init__方法中，主要有以下几个地方需要修改，均与从原来k1kxk1的3层卷积变为当前k1kx的2层卷积有关：
   1. 为了与k1kxk1结构保持一致的量化兼容性，需要将原来的nbitsA和nbitsW字段的量化精度长度验证由原来的3个变为2个

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665632342202-257a56da-0481-4c37-abcb-e9818f031cc6.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=148&id=u0ac6849e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=211&originWidth=862&originalType=binary&ratio=1&rotation=0&showTitle=false&size=46084&status=done&style=none&taskId=u157f60bd-e461-43eb-ac8f-f50c06c1c3e&title=&width=605)

   2. 创建卷积算子时，原来需要创建3个卷积，现在也只需要2个，且参数有所调整，第二个卷积输出维度直接变为整个block的输出维度

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665632416947-6cc5e04f-4e3e-4090-b1b7-f278fd2c0c74.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=276&id=u0f6327c3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=407&originWidth=887&originalType=binary&ratio=1&rotation=0&showTitle=false&size=106258&status=done&style=none&taskId=udfc5371e-ef2e-4a5a-9585-9c556fa0666&title=&width=601.5)

   3. 计算model_size和FLOPs时，也需要进行相应的调整，主要是将原来conv3相关的计算删除，且由于conv2的输出维度改变了，需要调整conv2后relu的FLOPs计算过程

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665632499354-f1c7df25-4a08-423e-9211-4b0f0b074e55.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=71&id=ueda92636&margin=%5Bobject%20Object%5D&name=image.png&originHeight=111&originWidth=972&originalType=binary&ratio=1&rotation=0&showTitle=false&size=33006&status=done&style=none&taskId=uf39c2bac-ab99-4b12-97d8-a4cbef94bc4&title=&width=622)

4. forward方法中，由于没有了conv3，因此仅需要将conv3相关的计算过程删除即可

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665632701414-f62907a8-331d-48e5-9548-39dd6448f32c.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=503&id=ue591b08a&margin=%5Bobject%20Object%5D&name=image.png&originHeight=683&originWidth=733&originalType=binary&ratio=1&rotation=0&showTitle=false&size=131346&status=done&style=none&taskId=ua4fc854d-63fe-4843-a5d5-bb256220c7a&title=&width=539.5)

5. get_model_size方法中，需要范围model_size列表的分支中需要将conv3相关的计算过程删除

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665632827464-a3c61342-72fe-4f6a-ad9b-26e423b1ef35.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=79&id=ub9b728da&margin=%5Bobject%20Object%5D&name=image.png&originHeight=127&originWidth=1165&originalType=binary&ratio=1&rotation=0&showTitle=false&size=29865&status=done&style=none&taskId=ue949369a-ee20-415f-86fe-d3b11485ec4&title=&width=724.5)

6. get_num_layers方法中，层数由原来的3层变为现在的2层；get_num_channels_list方法中，也只需要当前k1kx两层的输出维度

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665632896533-13f8e6f0-b979-40d8-bbac-c14192088878.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=43&id=u380aff73&margin=%5Bobject%20Object%5D&name=image.png&originHeight=58&originWidth=379&originalType=binary&ratio=1&rotation=0&showTitle=false&size=6118&status=done&style=none&taskId=ud2371eda-e7ed-4165-a64e-0a5281755a3&title=&width=280.5)

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665633047912-2035cc81-a951-45f0-98d2-f10aa06b58c4.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=44&id=u77d3bc90&margin=%5Bobject%20Object%5D&name=image.png&originHeight=66&originWidth=781&originalType=binary&ratio=1&rotation=0&showTitle=false&size=15413&status=done&style=none&taskId=u3d5fcc9d-1c45-4038-b578-62d460ad40b&title=&width=517.5)

7. get_log_zen_score方法中，将conv3相关的计算从中删除

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665633273120-dd6f0af0-5381-4072-bc8a-a4d72ef70d95.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=122&id=u5cd2b817&margin=%5Bobject%20Object%5D&name=image.png&originHeight=235&originWidth=1409&originalType=binary&ratio=1&rotation=0&showTitle=false&size=92060&status=done&style=none&taskId=u51e0b2dc-a98c-4cba-9deb-03da6b245ee&title=&width=732)

8. get_max_feature_num方法中，将conv3相关的计算从中删除，且由于此时conv2即将residual连接加和，所以conv2部分需要剔除residual_featmap项

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665633533038-cbdaba5d-369a-44d8-ac62-9d4a1500b556.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=152&id=ub5776cf7&margin=%5Bobject%20Object%5D&name=image.png&originHeight=234&originWidth=857&originalType=binary&ratio=1&rotation=0&showTitle=false&size=66851&status=done&style=none&taskId=u19c5cc82-2bfb-4184-8555-571b01c4033&title=&width=557.5)

9. 除了基础模块ResConvK1KX外，还需要将超模块的类名由SuperResConvK1KXK1改为SuperResConvK1KX，并将其中的inner_class修改为基础模块ResConvK1KX

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665633740994-278838b4-6b5c-4893-8a6f-b398153c4a1d.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=340&id=uf9207d24&margin=%5Bobject%20Object%5D&name=image.png&originHeight=525&originWidth=886&originalType=binary&ratio=1&rotation=0&showTitle=false&size=90823&status=done&style=none&taskId=u8ccbd9b3-7876-40af-b979-fed409af886&title=&width=573)

10. 最后，还需要将新增的模块添加到可用模块列表中才可以使用，主要需要修改2个地方：
    1. 在新的结构模块SuperResConvK1KXK1.py中，将最后__module_blocks__中的模块改为新增的k1kx模块

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665633925883-30686861-0ac6-4980-954b-1142822731f5.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=67&id=u30426356&margin=%5Bobject%20Object%5D&name=image.png&originHeight=97&originWidth=457&originalType=binary&ratio=1&rotation=0&showTitle=false&size=14693&status=done&style=none&taskId=u2c08fd40-48f9-49e9-b814-11852bc42ae&title=&width=316.5)

    2. 在__init__.py中，将新的结构模块导入并添加到__all_blocks__字典中

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665634096408-844fc86b-8268-4ca4-bd60-efdfb244d160.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=241&id=uaf2c2d24&margin=%5Bobject%20Object%5D&name=image.png&originHeight=377&originWidth=520&originalType=binary&ratio=1&rotation=0&showTitle=false&size=62443&status=done&style=none&taskId=u9e1df866-7110-423b-9a02-c3bdc187e3e&title=&width=333)

> 至此，新的k1kx结构模板已经添加完成。我们基于k1kxk1结构模板进行改造，主要修改了由于k1kxk1变为k1kx带来的层数减少和第二层输出维度变化相关的代码。最终，将新添加的k1kx结构模板添加到可用模块列表中。
> 
> 之后在使用时，在传入MasterNet的网络结构字符串中，如果模块的类别是ResConvK1KX或者SuperResConvK1KX，MasterNet就会自动根据k1kx结构模板生成对应的k1kx结构的网络模型。

<details>
<summary>完整nas/models/blocks/SuperResConvK1KX.py代码</summary>

```python
# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
import copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .blocks_basic import *


class ResConvK1KX(nn.Module):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''
        :param structure_info: {
            'class': 'ResConvK1KX',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn': bottleneck_channels,
            'nbitsA': input activation quant nbits list(default=8),
            'nbitsW': weight quant nbits list(default=8),
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
        self.stride = 1 if 's' not in structure_info else structure_info['s']
        self.bottleneck_channels = structure_info['btn']
        assert self.stride == 1 or self.stride == 2
        if "act" not in structure_info:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(structure_info['act']) 
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer

        if 'force_resproj' in structure_info:
            self.force_resproj = structure_info['force_resproj']
        else:
            self.force_resproj = False

        if "nbitsA" in structure_info and "nbitsW" in structure_info:
            self.quant = True
            self.nbitsA = structure_info["nbitsA"]
            self.nbitsW = structure_info["nbitsW"]
            if len(self.nbitsA)!=2 or len(self.nbitsW)!=2:
                raise ValueError("nbitsA/W must has 2 elements in %s, not nbitsA %d or nbitsW %d"%
                        (self.__class__, len(self.nbitsA), len(self.nbitsW)))
        else:
            self.quant = False

        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        if 'p' in structure_info:
            self.padding = structure_info['p']
        else:
            self.padding = (self.kernel_size - 1) // 2

        self.model_size = 0.0
        self.flops = 0.0

        self.block_list = []

        conv1_info = {'in': self.in_channels, 'out': self.bottleneck_channels, 'k': 1,
                            's': 1, 'g': self.groups, 'p': 0}
        conv2_info = {'in': self.bottleneck_channels, 'out': self.out_channels, 'k': self.kernel_size,
                            's': self.stride, 'g': self.groups, 'p': self.padding}
        if self.quant:
            conv1_info = {**conv1_info, **{"nbitsA":self.nbitsA[0], "nbitsW":self.nbitsW[0]}}
            conv2_info = {**conv2_info, **{"nbitsA":self.nbitsA[1], "nbitsW":self.nbitsW[1]}}

        self.conv1 = ConvKXBN(conv1_info, no_create=no_create, **kwargs)
        self.conv2 = ConvKXBN(conv2_info, no_create=no_create, **kwargs)

        # if self.no_create:
        #     pass
        # else:
        #     network_weight_stupid_bn_zero_init(self.conv3)

        self.block_list.append(self.conv1)
        self.block_list.append(self.conv2)

        self.model_size = self.model_size + self.conv1.get_model_size() + self.conv2.get_model_size()
        self.flops = self.flops + self.conv1.get_flops(1.0) + self.conv2.get_flops(1.0) \
            + self.bottleneck_channels + self.out_channels / self.stride ** 2  # add relu flops

        # residual link
        if self.stride == 2:
            if self.no_create:
                pass
            else:
                self.residual_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
            self.flops = self.flops + self.in_channels
        else:
            if self.no_create:
                pass
            else:
                self.residual_downsample = nn.Identity()

        if self.in_channels != self.out_channels or self.force_resproj:
            if self.quant:
                self.residual_proj = ConvKXBN({'in': self.in_channels, 'out': self.out_channels, 'k': 1,
                                           's': 1, 'g': 1, 'p': 0, "nbitsA":self.nbitsA[0], "nbitsW":self.nbitsW[0]}, no_create=no_create)
            else:
                self.residual_proj = ConvKXBN({'in': self.in_channels, 'out': self.out_channels, 'k': 1,
                                           's': 1, 'g': 1, 'p': 0}, no_create=no_create)
            self.model_size = self.model_size + self.residual_proj.get_model_size()
            self.flops = self.flops + self.residual_proj.get_flops(1.0 / self.stride) + self.out_channels / self.stride ** 2

            # if self.no_create:
            #     pass
            # else:
            #     network_weight_stupid_init(self.residual_proj)
        else:
            if self.no_create:
                pass
            else:
                self.residual_proj = nn.Identity()

    def forward(self, x, compute_reslink=True):
        reslink = self.residual_downsample(x)
        reslink = self.residual_proj(reslink)

        output = x
        output = self.conv1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        output = self.activation_function(output)
        output = self.conv2(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        if self.dropout_layer is not None:
            if np.random.rand() <= self.dropout_layer:
                output = 0 * output + reslink
            else:
                output = output + reslink
        else:
            output = output + reslink

        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        output = self.activation_function(output)

        return output
        

    def get_model_size(self, return_list=False):
        if return_list:
            return self.conv1.get_model_size(return_list)+self.conv2.get_model_size(return_list)
        else:
            return self.model_size


    def get_flops(self, resolution):
        return self.flops * resolution**2


    def get_num_layers(self):
        return 2
        

    def get_output_resolution(self, input_resolution):
        resolution = input_resolution
        for block in self.block_list:
            resolution = block.get_output_resolution(resolution)
        return resolution
        
        
    def get_params_for_trt(self, input_resolution):
        # generate the params for yukai's predictor
        params = []
        the_res = input_resolution
        for idx, block in enumerate(self.block_list):
            if self.residual_proj and idx==len(self.block_list)-1:
                params_temp = block.get_params_for_trt(the_res, elmtfused=1) # if reslink, elmtfused=1 
            else:
                params_temp = block.get_params_for_trt(the_res)
            the_res = block.get_output_resolution(the_res)
            params += params_temp
        if isinstance(self.residual_proj, ConvKXBN):
            params_temp = self.residual_proj.get_params_for_trt(the_res)
            params += params_temp

        return params


    def entropy_forward(self, x, skip_relu=True, skip_bn=True, **kwarg):
        output = x
        output_std_list = []
        output_std_block = 1.0
        for the_block in self.block_list:
            output = the_block(output, skip_bn=skip_bn)
            if not skip_relu: output = self.activation_function(output)
            # print("output std: mean %.4f, std %.4f, max %.4f, min %.4f\n"%(
                    # output.mean().item(), output.std().item(), output.max().item(), output.min().item()))
            if "init_std_act" in kwarg and hasattr(self, "nbitsA"):
                output_std_block *= output.std()/kwarg["init_std_act"]
                output = output/(output.std()/kwarg["init_std_act"])
            else:
                output_std_block *= output.std()
                output = output/output.std()
        output_std_list.append(output_std_block)
        return output, output_std_list


    def get_num_channels_list(self):
        return [self.bottleneck_channels, self.out_channels]


    def get_log_zen_score(self, **kwarg):
        if "init_std" in kwarg and "init_std_act" in kwarg and hasattr(self, "nbitsA"):
            conv1_std = np.log(STD_BITS_LUT[kwarg["init_std_act"]][self.nbitsA[0]]*STD_BITS_LUT[kwarg["init_std"]][self.nbitsW[0]])-np.log(kwarg["init_std_act"])
            conv2_std = np.log(STD_BITS_LUT[kwarg["init_std_act"]][self.nbitsA[1]]*STD_BITS_LUT[kwarg["init_std"]][self.nbitsW[1]])-np.log(kwarg["init_std_act"])

            return [np.log(np.sqrt(self.in_channels)) + conv1_std + \
                    np.log(np.sqrt(self.bottleneck_channels * self.kernel_size ** 2)) + conv2_std]
        else:
            return [np.log(np.sqrt(self.in_channels)) + \
                    np.log(np.sqrt(self.bottleneck_channels * self.kernel_size ** 2))]

    def get_max_feature_num(self, resolution):
        residual_featmap = resolution**2*self.out_channels//(self.stride**2)
        if self.quant:
            residual_featmap = residual_featmap * self.nbitsA[0] / 8
        conv1_max_featmap = self.conv1.get_max_feature_num(resolution) + residual_featmap
        conv2_max_featmap = self.conv2.get_max_feature_num(resolution)
        max_featmap_list = [conv1_max_featmap, conv2_max_featmap]

        return max_featmap_list


class SuperResConvK1KX(BaseSuperBlock):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'SuperResConvK1KX',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
        }
        :param NAS_mode:
        '''
        structure_info['inner_class'] = 'ResConvK1KX'
        super().__init__(structure_info=structure_info, no_create=no_create, inner_class=ResConvK1KX,
                         dropout_channel=dropout_channel, dropout_layer=dropout_layer,
                         **kwargs)


__module_blocks__ = {
    'ResConvK1KX': ResConvK1KX,
    'SuperResConvK1KX': SuperResConvK1KX,
}
```
</details>


## 搜索空间添加

### 搜索空间基本流程

- 搜索空间位于nas/spaces目录下
- 单个搜索空间中最主要的函数为mutate_function，进化算法每次迭代时，会从当前网络结构集合中随机挑选一个网络，然后随机对其中的几个block进行“变异”以得到新的结构，此时就会对要变异的block调用该函数
- 每次每个block变异会根据block类型、随机选择的变异方法等因素进行变异
   - 以原k1kxk1搜索空间space_K1KXK1.py为例
   - 在搜索空间最开始的地方规定了其可能对哪些结构超参进行变异，以及每次变异可能的变异方法有哪些
      - 可变异超参：stem block只对输出维度out进行变异，其它block则可以对输出维度out、bottleneck卷积核大小k和维度btn、block中基础模块层数L进行变异
      - 变异方法：
         - 对于卷积核大小kernel_size，只在3或者5之间随机选择
         - 对于层数layer，每次变异可在当前层数基础上减少/增加1层或者2层
         - 对于维度channel，每次变异可在当前维度基础上进行指定倍数的变化

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665642303488-7eb50451-751f-453c-a624-53c7c15a1904.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=154&id=u5b27a4c3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=213&originWidth=626&originalType=binary&ratio=1&rotation=0&showTitle=false&size=43305&status=done&style=none&taskId=u7ed4af42-6664-4419-8253-7b13b48a601&title=&width=452)

   - 此外，各个搜索空间还可以在基本变异方法基础上添加一些用户自定义的限制条件，以进行更定制化搜索空间流程
      - 如上图中11行和15行分别限制了特征维度最大只能到2048，而btn相比于out最小不能小于out的1/10
      - 如下图中所示，在对每个block的基础模块层数L进行变异时，如果可以限制靠前的block中层数的数目，以将层数主要集中在模型更深的位置；并且else分支中则促使了在总层数限制范围内，深层的各block中的层数L分配更为平均

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665642724647-29e43e7f-5260-43b1-960b-b3fd733dfef9.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=115&id=ua0cb33e7&margin=%5Bobject%20Object%5D&name=image.png&originHeight=168&originWidth=790&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28663&status=done&style=none&taskId=u80e96aeb-59d0-4252-9cb6-9ac3e509c87&title=&width=539)

   - 用户可根据自身的具体需求、针对任务类型、网络结构差异等要素自行调整或者自定义搜索空间变异流程，以使得搜索出的网络结构能在不同场景表现出最佳性能
- 这里我们同样基于k1kxk1的搜索空间（即nas/spaces/space_K1KXK1.py）改造出k1kx的搜索空间

### k1kx搜索空间添加过程

1. 在nas/spaces目录下，复制space_K1KXK1.py为新的模板space_K1KX.py
2. 针对k1kx结构模板的搜索空间中，需要将针对的block类型改为前面添加的SuperResConvK1KX

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665647114403-56249f60-9e8e-40d3-905e-50247c8f9c31.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=77&id=u415f32f1&margin=%5Bobject%20Object%5D&name=image.png&originHeight=100&originWidth=797&originalType=binary&ratio=1&rotation=0&showTitle=false&size=19298&status=done&style=none&taskId=udfee8e57-5b89-43d3-9a9b-8c5b025e158&title=&width=615.5)

3. 此外，由于每个block的层数由3层变为2层，需要在L变异过程中将每个block层数限制中的3变为2；且由于针对DAMO-YOLO的backbone总层数较少，可进一步减少浅层block的层数

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665649466513-98f9fccd-87c1-44b9-8440-247133172fce.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=135&id=uaca67fc8&margin=%5Bobject%20Object%5D&name=image.png&originHeight=177&originWidth=796&originalType=binary&ratio=1&rotation=0&showTitle=false&size=34582&status=done&style=none&taskId=ue59fc847-53cd-4d3d-a8bf-3b1f69a5c73&title=&width=607)

4. 由于我们针对的是用于DAMO-YOLO的网络结构，因此在搜索时，需要进行若干自定义限制与调整
   1. 为了平衡FLOPs与实际运行速度，我们只需要k=3，即恒定保持k1k3的结构，因此将k从可变异超参列表中去除（注：此处假设k1kx的初始结构中所有block的k已经为3，具体可见下一节搜索脚本部分）

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665647752202-02f93d12-ed80-4e4f-b13f-e5eb2aba4e9d.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=75&id=u3001c5b8&margin=%5Bobject%20Object%5D&name=image.png&originHeight=105&originWidth=469&originalType=binary&ratio=1&rotation=0&showTitle=false&size=13566&status=done&style=none&taskId=u836fa574-9b62-45cb-927b-2f800495c73&title=&width=334.5)

   2. 扩充维度channel变异方法，使其更加对称

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665647936490-4630dc8a-e15d-4bca-815b-5f92702412a0.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=92&id=u2bb3169d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=127&originWidth=610&originalType=binary&ratio=1&rotation=0&showTitle=false&size=23245&status=done&style=none&taskId=u43f2efb0-ec1b-410e-a115-4204cb02d37&title=&width=443)

   3. 对于ConvKXBNRELU模块，添加限制：当前block输出维度不得大于下一个block输出维度

    ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665648321809-b48386f5-93d5-4091-83ec-e5467604bab1.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=217&id=uc3c713f1&margin=%5Bobject%20Object%5D&name=image.png&originHeight=321&originWidth=733&originalType=binary&ratio=1&rotation=0&showTitle=false&size=56097&status=done&style=none&taskId=ub58cd21d-a329-4fc4-a281-a3b03d6496e&title=&width=496.5)

   4. 对于SuperResConvK1KX模块的输出维度out的变异方法进行全面修改
      1. 针对需要搜索的模型大小不同，限制不同block的输出维度范围，以small级别为例，限制最后3个stage，即第3，5，6个block的输出维度out范围；在最前面变异方法处新增如下图所示的维度范围列表（注：此处假设k1kx的初始结构模仿k1kxk1的初始结构，在倒数第二个stage设置2个block，使其获得更多的层数以提高模型性能，具体可见下一节搜索脚本部分）

        ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665648724150-d4bb4d5e-09d9-4628-bfde-47b1536c307c.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=145&id=u2b91c4e8&margin=%5Bobject%20Object%5D&name=image.png&originHeight=201&originWidth=713&originalType=binary&ratio=1&rotation=0&showTitle=false&size=37589&status=done&style=none&taskId=ue0adb752-c0a9-4877-b96a-017f6e1ca50&title=&width=514.5)   

      2. 添加限制：输出维度必须不小于输入维度
      3. 添加限制：当前block输出维度必须不大于下一个block输出维度
      4. 由于新的输出维度同时是下一个block的输入维度，因此根据新的out调整下一层bottleneck维度btn
      5. out变异方法部分完整修改见下图
      
        ![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665649119973-436fede2-3b01-4685-93a5-915f591f9d4e.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=229&id=uf3de0c78&margin=%5Bobject%20Object%5D&name=image.png&originHeight=328&originWidth=783&originalType=binary&ratio=1&rotation=0&showTitle=false&size=84940&status=done&style=none&taskId=u0c6c9018-687c-4fc8-a1e7-31348422427&title=&width=545.5)

> 此时，一个较为完整的针对k1kx结构模板的搜索空间也建立完成了。我们在k1kxk1搜索空间的基础上，针对DAMO-YOLO的需求与k1kx结构的特点，做出了若干个部分的适应性调整。
> 
> 之后在使用时，我们会将该搜索空间的名称在搜索脚本中指定，并作为参数传递到nas框架中，nas框架会自动载入搜索空间并调用对应的mutate_function对网络结构进行变异。

<details>
<summary>完整nas/spaces/space_K1KX.py代码</summary>

```python
# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os, sys
import random, copy, bisect


stem_mutate_method_list = ['out']
mutate_method_list = ['out', 'btn', 'L']
# last_block_mutate_method_list = ['btn', 'L']

# add channel limit range for 6 blocks (only valid for No. 3, 5, 6 layers)
channel_range = [None, None, [64, 128], None, [128, 256], [256, 512]]
the_maximum_channel = 2048
search_kernel_size_list = [3, 5]
search_layer_list = [-2, -1, 1, 2]
search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5]
btn_minimum_ratio = 10 # the bottleneck must be larger than out/10

def smart_round(x, base=8):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)


def mutate_channel(channels):
    scale = random.choice(search_channel_list)
    new_channels = smart_round(scale*channels)
    new_channels = min(the_maximum_channel, new_channels)
    return new_channels


def mutate_kernel_size(kernel_size):
    for i in range(len(search_kernel_size_list)):
        new_kernel_size = random.choice(search_kernel_size_list)
        if new_kernel_size!=kernel_size:
            break
    return new_kernel_size


def mutate_layer(layer):
    for i in range(len(search_layer_list)):
        new_layer = layer + random.choice(search_layer_list)
        new_layer = max(1, new_layer)
        if new_layer!=layer:
            break
    return new_layer


def mutate_function(block_id, structure_info_list, budget_layers, minor_mutation=False):

    structure_info = structure_info_list[block_id]
    if block_id < len(structure_info_list)-1: 
        structure_info_next = structure_info_list[block_id+1]
    structure_info = copy.deepcopy(structure_info)
    class_name = structure_info['class']

    if class_name == 'ConvKXBNRELU':
        if block_id <= len(structure_info_list) - 2:
            random_mutate_method = random.choice(stem_mutate_method_list)
        else:
            return False

        if random_mutate_method == 'out':
            new_out = mutate_channel(structure_info['out'])
            # Add the constraint: the maximum output of the stem block is 128
            new_out = min(32, new_out)
            if block_id < len(structure_info_list)-1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            return [structure_info]

        if random_mutate_method == 'k':
            new_k = mutate_kernel_size(structure_info['k'])
            structure_info['k'] = new_k
            return [structure_info]

    elif class_name == 'SuperResConvK1KX':
        # coarse2fine mutation flag, only mutate the channels' output
        mutate_method_list_final=['out', 'btn'] if minor_mutation else mutate_method_list

        random_mutate_method = random.choice(mutate_method_list_final)

        if random_mutate_method == 'out':
            new_out = mutate_channel(structure_info['out'])
            # Add the contraint: output_channel should be in range [min, max]
            if channel_range[block_id] is not None:
                this_min, this_max = channel_range[block_id]
                new_out = max(this_min, min(this_max, new_out))
            # Add the constraint: output_channel > input_channel
            new_out = max(structure_info['in'], new_out)
            if block_id < len(structure_info_list) - 1:
                new_out = min(structure_info_next['out'], new_out)
            structure_info['out'] = new_out
            if block_id < len(structure_info_list) - 1 and "btn" in structure_info_next:
                new_btn = min(new_out, structure_info_next['btn'])
                structure_info_next['btn'] = new_btn

        if random_mutate_method == 'k':
            new_k = mutate_kernel_size(structure_info['k'])
            structure_info['k'] = new_k

        if random_mutate_method == 'btn':
            new_btn = mutate_channel(structure_info['btn'])
            # Add the constraint: bottleneck_channel <= output_channel
            new_btn = min(structure_info['out'], new_btn) 
            structure_info['btn'] = new_btn

        if random_mutate_method == 'L':
            new_L = mutate_layer(structure_info['L'])
            # add the constraint: the block 1 can't have the large layers.
            if block_id==1:
                new_L = min(2, new_L)
            else:
                new_L = min(int(budget_layers//2//(len(structure_info_list)-2)), new_L)

            structure_info['L'] = new_L

        # add the constraint: the btn must be larger than out/btn_minimum_ratio.
        if structure_info['btn']<(structure_info['out']/btn_minimum_ratio):
            structure_info['btn'] = smart_round(structure_info['out']/btn_minimum_ratio)
        
        return [structure_info]
    
    else:
        raise RuntimeError('Not implemented class_name=' + class_name)
```
</details>


## 搜索脚本添加

### 搜索脚本基本内容

- 可参考scripts目录下各种类型的搜索脚本，参数列表可参考configs/config_nas.py
- 基本包含4个部分：
   - 可调参数设定：可以挑选若干重要可调参数放置于脚本前，便于通过脚本参数进行控制
   - 工作空间准备：根据参数信息自行组装工作空间名称与路径，并将搜索空间、当前搜索脚本等搜索任务元信息拷贝到工作空间
   - 初始结构准备：设定搜索初始结构，定义各个block的初始超参数；如果包括混合精度量化，还需预定义各层的初始量化精度
   - 搜索代码运行：通过mpirun进行多进程异步协同搜索，调用搜索代码，并装填各个可调参数
- 由于DAMO-YOLO针对的是检测任务，而且暂时不需要考虑量化，因此我们可基于scripts/detection中的搜索脚本进行改造
- 由于我们要考虑模型的速度，因此选择带有latency预测的脚本scripts/detection/example_R50_predictor.sh进行改造，改造目标以small级别为例

### 添加过程

1. 在scripts目录下新建一个目录damo-yolo，用于专门存放我们的新的搜索脚本；复制scripts/detection/example_R50_predictor.sh到scripts/damo-yolo/example_k1kx_small.sh
2. 原脚本较为简单，为了方便管理，我们将部分重要可调参数移动到最前方。这里包括5个可调参数
   1. image_size和image_channel：输入网络的图像的大小和通道数
      1. 由于在实际使用时，我们会将stem层替换为Focus结构，因此虽然这里分别设置为320和12，但是实际对应的原图大小为640，通道数为3（更详细初始结构可见下面初始结构的部分）
   2. layers：搜索时总卷积层数限制。small级别对标backbone有20多层，因此我们设置25层的限制
   3. latency：搜索时的latency限制。搜索时模型的latency通过预置的一个算子latency数据库预测得到，因此与真实值有偏差，在设定前需要进行若干实验进行一个尺度对齐，这里设置为0.12ms，即12e-5
   4. last_ratio：搜索时最后一个stage的分数权重。计算模型得分时，对各个stage的重视程度是不同的，通常来说对最后一个stage需要设置更大的阈值，这里设置默认为16

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665660250193-0e123980-3eea-4f84-b4ed-7659b45910c4.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=223&id=u5a5bf900&margin=%5Bobject%20Object%5D&name=image.png&originHeight=286&originWidth=318&originalType=binary&ratio=1&rotation=0&showTitle=false&size=21376&status=done&style=none&taskId=u75f616a4-00d6-4c3c-83ef-d323f157c49&title=&width=248)

3. 工作空间准备：
   1. 将工作空间名称用可调参数进行命名以区分不同搜索实验，并单独用一个工作目录存放damo-yolo系列搜索结果
   2. 修改搜索空间为space_K1KX
   3. 更新需要拷贝的该搜索脚本的路径

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665660315226-ec36be7b-587c-456b-9ce6-5b9cf0a6cd06.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=144&id=u612f6fb5&margin=%5Bobject%20Object%5D&name=image.png&originHeight=199&originWidth=658&originalType=binary&ratio=1&rotation=0&showTitle=false&size=38687&status=done&style=none&taskId=u9d1633f5-034b-4058-9c49-b60f4f8cd79&title=&width=477)

4. 初始结构准备：由于要求初始结构必须在搜索限制范围内，因此初始结构可以设定一个较小的网络
   1. 注意到stride为2的block仅有4个，因此虽然共有6个block，但是总共有5个stage，倒数第2个stage包含2个block
   2. 正如上文所说，第一个stage的conv在最终使用时会替换为Focus结构，因此其stride为1，且输入通道维度设置为12
   3. 我们将所有block的k1kx的卷积核大小设置为3，再由于k1kx搜索空间中不会对k进行变异，因此最终搜索结构也肯定为k1k3结构

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665660345113-b1734c69-009e-4c73-82a1-e9e146f8f707.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=183&id=u00c1d73e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=237&originWidth=789&originalType=binary&ratio=1&rotation=0&showTitle=false&size=50634&status=done&style=none&taskId=ube2a43da-d1cd-4835-b086-c4a74a091b5&title=&width=608.5)

5. 搜索代码运行：最后使用mpirun运行搜索代码，可以将上面的预设可调参数变量填充到运行时参数位置，并且做出其他的一些改进：
   1. 分数计算方式改为madnas而非entropy，这样只需要cpu而不需要gpu，同时搜索进程数目可增大到64（可按照自己的机器能力调节）
   2. 添加对budget和score的image_channel的参数设置
   3. 针对DAMO-YOLO的latency评估，LightNAS提供了一个单独的测速数据库，因此需要将测速数据类型改为专门的类型“FP16_DAMOYOLO”（目前针对DAMOYOLO暂时仅支持FP16）
   4. 适当增大进化算法的网络集合大小、迭代次数和log打印间隔次数

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1669186731115-63c847a6-ead9-426e-a9f8-b415ef508422.png#clientId=u0f70af76-d2b8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=212&id=u8b27e523&margin=%5Bobject%20Object%5D&name=image.png&originHeight=290&originWidth=1104&originalType=binary&ratio=1&rotation=0&showTitle=false&size=131589&status=done&style=none&taskId=u8615d40a-0368-44e7-ad83-e9ab389544b&title=&width=806)

> 此时，一个较为完整的针对k1kx结构模板的搜索空间也建立完成了。我们在k1kxk1搜索空间的基础上，针对DAMO-YOLO的需求与k1kx结构的特点，做出了若干个部分的适应性调整。
> 
> 之后在使用时，我们会将该搜索空间的名称在搜索脚本中指定，并作为参数传递到nas框架中，nas框架会自动载入搜索空间并调用对应的mutate_function对网络结构进行变异。

<details>
<summary>完整scripts/damo-yolo/example_k1kx_small.sh代码</summary>

```shell
cd "$(dirname "$0")"
set -e

cd ../../

# 可调参数设定
image_size=320
image_channel=12

layers="${1:-"25"}"
latency="${2:-"12e-5"}"
last_ratio="${3:-"16"}"

# 工作空间准备
name=k1kx_R640__layers${layers}_lat${latency}_ratio${last_ratio}
work_dir=save_model/LightNAS/damo-yolo/${name}

mkdir -p ${work_dir}
space_mutation="space_K1KX"
cp nas/spaces/${space_mutation}.py ${work_dir}
cp scripts/damo-yolo/example_k1kx_small.sh ${work_dir}

echo "[\
{'class': 'ConvKXBNRELU', 'in': 12, 'out': 32, 's': 1, 'k': 3}, \
{'class': 'SuperResConvK1KX', 'in': 32, 'out': 48, 's': 2, 'k': 3, 'L': 1, 'btn': 32}, \
{'class': 'SuperResConvK1KX', 'in': 48, 'out': 96, 's': 2, 'k': 3, 'L': 1, 'btn': 48}, \
{'class': 'SuperResConvK1KX', 'in': 96, 'out': 96, 's': 2, 'k': 3, 'L': 1, 'btn': 96}, \
{'class': 'SuperResConvK1KX', 'in': 96, 'out': 192, 's': 1, 'k': 3, 'L': 1, 'btn': 96}, \
{'class': 'SuperResConvK1KX', 'in': 192, 'out': 384, 's': 2, 'k': 3, 'L': 1, 'btn': 192}, \
]" \
  >${work_dir}/init_structure.txt

rm -rf acquired_gpu_list.*
mpirun --allow-run-as-root -np 64 -H 127.0.0.1:64 -bind-to none -map-by slot -mca pml ob1 \
  -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python nas/search.py configs/config_nas.py --work_dir ${work_dir} \
  --cfg_options task="detection" space_classfication=False only_master=False log_level="INFO" \
  budget_image_size=${image_size} budget_flops=None budget_latency=${latency} budget_layers=${layers} budget_model_size=None \
  score_type="madnas" score_batch_size=32 score_image_size=${image_size} score_repeat=4 score_multi_ratio=[0,0,1,1,${last_ratio}] \
  budget_image_channel=${image_channel} score_image_channel=${image_channel} \
  lat_gpu=False lat_pred=True lat_date_type="FP16_DAMOYOLO" lat_pred_device="V100" lat_batch_size=32 \
  space_arch="MasterNet" space_mutation=${space_mutation} space_structure_txt=${work_dir}/init_structure.txt \
  ea_popu_size=512 ea_log_freq=5000 ea_num_random_nets=500000
```
</details>

## 其他细节代码调整

- 由于在搜索的主流程代码nas/search.py中，会对网络结构的层数做一个初始的计算与判断，它是靠每个block的类别来进行层数计算的，因此需要在那里添加上新增的k1kx的结构模板

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/26556813/1665650924550-57c7215a-b2cc-4865-a5a8-c18d39ed51fe.png#clientId=uf122ab90-842b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=116&id=u717fd2e9&margin=%5Bobject%20Object%5D&name=image.png&originHeight=150&originWidth=625&originalType=binary&ratio=1&rotation=0&showTitle=false&size=25952&status=done&style=none&taskId=u4e507baf-c8ec-4c38-b03c-1b11ee7d3f8&title=&width=481.5)

- 最后，就可以按照[DAMO-YOLO backbone结构搜索](#DAMO-YOLO backbone结构搜索)章节的内容进行搜索了
