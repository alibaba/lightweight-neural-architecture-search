# TinyNAS
                                                                  
- This repository is a collection of training-free neural architecture search methods developed by TinyML team, Data Analytics and Intelligence Lab, Alibaba DAMO Academy. Researchers and developers can use this toolbox to design their neural architectures with different budgets on CPU devices within 30 minutes.
    - Training-Free Neural Architecture Evaluation Scores by Entropy [**DeepMAD**](https://arxiv.org/abs/2303.02165)(CVPR'23), and by Gradient [**Zen-NAS**](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Zen-NAS_A_Zero-Shot_NAS_for_High-Performance_Image_Recognition_ICCV_2021_paper.pdf)(ICCV'21)
    - Joint Quantization and Architecture Search [**Mixed-Precision Quantization Search**](https://openreview.net/pdf?id=lj1Eb1OPeNw)(NeurIPS'22)
    - Application : Object Detection [**MAE-DET**](https://proceedings.mlr.press/v162/sun22c/sun22c.pdf)(ICML'22)
    - Application : Action Recognition [**Maximizing Spatio-Temporal Entropy**](https://openreview.net/pdf?id=lj1Eb1OPeNw)(ICLR'23)

## News

- **:sunny: Hiring research interns for Neural Architecture Search, Tiny Machine Learning, Computer Vision tasks: [xiuyu.sxy@alibaba-inc.com](xiuyu.sxy@alibaba-inc.com)**
- :boom: 2023.04: [**PreNAS: Preferred One-Shot Learning Towards Efficient Neural Architecture Search**](https://arxiv.org/abs/2304.14636) is accepted by ICML'23.
- :boom: 2023.04: We will give a talk on Zero-Cost NAS at [**IFML Workshop**](https://www.ifml.institute/events/ifml-workshop-2023), April 20, 2023.
- :boom: 2023.03: Code for [**E3D**](configs/action_recognition/README.md) is now released.
- :boom: 2023.03: The code is refactoried and DeepMAD is supported.
- :boom: 2023.03: [**DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network**](https://arxiv.org/abs/2303.02165) is accepted by CVPR'23.
- :boom: 2023.02: A demo is available on [**ModelScope**](https://modelscope.cn/studios/damo/TinyNAS/summary)
- :boom: 2023.01: [**Maximizing Spatio-Temporal Entropy of Deep 3D CNNs for Efficient Video Recognition**](https://openreview.net/pdf?id=lj1Eb1OPeNw) is accepted by ICLR'23.
- :boom: 2022.11: [**DAMO-YOLO**](https://github.com/tinyvision/DAMO-YOLO) backbone search is now supported! And paper is on [ArXiv](https://arxiv.org/abs/2211.15444) now.
- :boom: 2022.09: [**Mixed-Precision Quantization Search**](configs/quant/README.md) is now supported! The [**QE-Score**](https://openreview.net/pdf?id=E28hy5isRzC) paper is accepted by NeurIPS'22.
- :boom: 2022.08: We will give a tutorial on [**Functional View for Zero-Shot NAS**](https://mlsys.org/virtual/2022/tutorial/2201) at MLSys'22.
- :boom: 2022.06: Code for [**MAE-DET**](configs/detection/README.md) is now released.
- :boom: 2022.05: [**MAE-DET**](https://proceedings.mlr.press/v162/sun22c/sun22c.pdf) is accepted by ICML'22.
- :boom: 2021.09: Code for [**Zen-NAS**](https://github.com/idstcv/ZenNAS) is now released.
- :boom: 2021.07: The inspiring training-free paper [**Zen-NAS**](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Zen-NAS_A_Zero-Shot_NAS_for_High-Performance_Image_Recognition_ICCV_2021_paper.pdf) has been accepted by ICCV'21.

## Features

- This toolbox consists of multiple modules including the following :
    - [Search Searcher module](tinynas/searchers/README.md)
    - [Search Strategy module](tinynas/strategy/README.md)
    - [Models Definition module](tinynas/models/README.md)
    - [Score module](tinynas/scores/README.md)
    - [Search Space module](tinynas/spaces/README.md)
    - [Budgets module](tinynas/budgets/README.md)
    - [Latency Module](tinynas/latency/op_profiler/README.md)
    - [Population module](tinynas/evolutions/README.md)

It manages these modules with the help of [ModelScope](https://github.com/modelscope/modelscope) Registry and Configuration mechanism.

- The `Searcher` is defined to be responsible for building and completing the entire search process. Through the combination of these modules and the corresponding configuration files, we can complete backbone search for different tasks (such as classification, detection, etc.) under different budget constraints (such as the number of parameters, FLOPs, delay, etc.).

- Currently supported tasks: For each task, we provide several sample configurations and scripts as follows to help you get started quickly.

    - `Classification`：Please Refer to [Search Space](tinynas/spaces/space_k1kxk1.py) and [Config](configs/classification/R50_FLOPs.py)
    - `Detection`：Please Refer to [Search Space](tinynas/spaces/space_k1kx.py) and [Config](configs/detection/R50_FLOPs.py)
    - `Quantization`: Please Refer to [Search Space](tinynas/spaces/space_quant_k1dwk1.py) and [Config](configs/quant/Mixed_7d0G.py)

***
## Installation
- Please Refer to [installation.md](installation.md)

***
## How to Use
- Please Refer to [get_started.md](get_started.md)

***
## Results
### Results for Classification（[Details](configs/classification/README.md)）

|Backbone|Param (MB)|FLOPs (G)|ImageNet TOP1|Structure|Download|
|:----|:----|:----|:----|:----|:----|
|DeepMAD-R18|11.69|1.82|77.7%| [txt](configs/classification/models/deepmad-R18.txt)|[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-R18/R18.pth.tar)|
|DeepMAD-R34|21.80|3.68|79.7%| [txt](configs/classification/models/deepmad-R34.txt)|[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-R18/R34.pth.tar) |
|DeepMAD-R50|25.55|4.13|80.6%|[txt](configs/classification/models/deepmad-R50.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-R18/R50.pth.tar) |
|DeepMAD-29M-224|29|4.5|82.5%|[txt](configs/classification/models/deepmad-29M-224.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-29M-224/DeepMAD-29M-Res224-82.5acc.pth.tar) |
|DeepMAD-29M-288|29|4.5|82.8%|[txt](configs/classification/models/deepmad-29M-288.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-29M-288/DeepMAD-29M-Res288-82.8acc.pth.tar) |
|DeepMAD-50M|50|8.7|83.9%|[txt](configs/classification/models/deepmad-50M.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-50M/DeepMAD-50M-Res224-83.9acc.pth.tar) |
|DeepMAD-89M|89|15.4|84.0%|[txt](configs/classification/models/deepmad-89M.txt) |[model](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-89M/DeepMAD-89M-Res224-84.0acc.pth.tar) |                                                   
| Zen-NAS-R18-like | 10.8 |    1.7     |   78.44  | [txt](configs/classification/models/R18-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R18-like.pth.tar) |
| Zen-NAS-R50-like | 21.3 |    3.6     |   80.04  | [txt](configs/classification/models/R50-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R50-like.pth.tar) |
| Zen-NAS-R152-like | 53.5 |    10.5     |   81.59  | [txt](configs/classification/models/R152-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R152-like.pth.tar) |
> The official code for **Zen-NAS** was originally released at https://github.com/idstcv/ZenNAS.   <br/>
                                                                                                                        
***
### Results for low-precision backbones（[Details](configs/quant/README.md)）

|Backbone|Param (MB)|BitOps (G)|ImageNet TOP1|Structure|Download|
|:----|:----|:----|:----|:----|:----|
|MBV2-8bit|3.4|19.2|71.90%| -| -|
|MBV2-4bit|2.3|7|68.90%| -|- |
|Mixed19d2G|3.2|18.8|74.80%|[txt](configs/quant/models/mixed7d0G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-7d0G/quant_238_70.7660.pth.tar) |
|Mixed7d0G|2.2|6.9|70.80%|[txt](configs/quant/models/mixed19d2G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-19d2G/quant_237_74.8180.pth.tar) |
                                                                              
***
### Results for Object Detection（[Details](configs/detection/README.md)）
| Backbone | Param (M) | FLOPs (G) |   box AP<sub>val</sub> |   box AP<sub>S</sub> |   box AP<sub>M</sub>  |   box AP<sub>L</sub> | Structure | Download |
|:---------:|:---------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:------:|
| ResNet-50 | 23.5 |    83.6    |  44.7 | 29.1 | 48.1 | 56.6  | - | - |
| ResNet-101| 42.4 |    159.5   |  46.3 | 29.9 | 50.1 | 58.7  | - | - |
| MAE-DET-S | 21.2 |    48.7    |  45.1 | 27.9 | 49.1 | 58.0  | [txt](configs/detection/models/maedet_s.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-s/latest.pth) |
| MAE-DET-M | 25.8 |    89.9    |  46.9 | 30.1 | 50.9 | 59.9  | [txt](configs/detection/models/maedet_m.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-m/latest.pth) |
| MAE-DET-L | 43.9 |    152.9   |  47.8 | 30.3 | 51.9 | 61.1  | [txt](configs/detection/models/maedet_l.txt)      |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-l/latest.pth) |

***
## Results for Action Recognition ([Details](configs/action_recognition/README.md)）

| Backbone  | size   |  FLOPs (G) |  SSV1 Top-1 | SSV1 Top-5 | Structure | 
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| X3D-S | 160 |    1.9     |   44.6  | 74.4| -     |
| X3D-S | 224 |    1.9     |   47.3  | 76.6| -     |
| E3D-S | 160 |    1.9     |   47.1  | 75.6| [txt](configs/action_recognition/models/E3D_S.txt)       |
| E3D-M  | 224 |     4.7     |   49.4  | 78.1| [txt](configs/action_recognition/models/E3D_M.txt)       |
| E3D-L  | 312 |     18.3     |   51.1  | 78.7| [txt](configs/action_recognition/models/E3D_L.txt)       |

***
**Note**：
If you find this useful, please support us by citing them.
```
@inproceedings{cvpr2023deepmad,
	title = {DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network},
	author = {Xuan Shen and Yaohua Wang and Ming Lin and Yilun Huang and Hao Tang and Xiuyu Sun and Yanzhi Wang},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2023},
	url = {https://arxiv.org/abs/2303.02165}
}

@inproceedings{icml23prenas,
	title={PreNAS: Preferred One-Shot Learning Towards Efficient Neural Architecture Search},
	author={Haibin Wang and Ce Ge and Hesen Chen and Xiuyu Sun},
	booktitle={International Conference on Machine Learning},
	year={2023},
	organization={PMLR}
}

@inproceedings{iclr23maxste,
	title     = {Maximizing Spatio-Temporal Entropy of Deep 3D CNNs for Efficient Video Recognition},
	author    = {Junyan Wang and Zhenhong Sun and Yichen Qian and Dong Gong and Xiuyu Sun and Ming Lin and Maurice Pagnucco and Yang Song },
	journal   = {International Conference on Learning Representations},
	year      = {2023},
}

@inproceedings{neurips23qescore,
	title     = {Entropy-Driven Mixed-Precision Quantization for Deep Network Design},
	author    = {Zhenhong Sun and Ce Ge and Junyan Wang and Ming Lin and Hesen Chen and Hao Li and Xiuyu Sun},
	journal   = {Advances in Neural Information Processing Systems},
	year      = {2022},
}

@inproceedings{icml22maedet,
	title={MAE-DET: Revisiting Maximum Entropy Principle in Zero-Shot NAS for Efficient Object Detection},
	author={Zhenhong Sun and Ming Lin and Xiuyu Sun and Zhiyu Tan and Hao Li and Rong Jin},
	booktitle={International Conference on Machine Learning},
	year={2022},
	organization={PMLR}
}

@inproceedings{iccv21zennas,
	title     = {Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition},
	author    = {Ming Lin and Pichao Wang and Zhenhong Sun and Hesen Chen and Xiuyu Sun and Qi Qian and Hao Li and Rong Jin},
	booktitle = {2021 IEEE/CVF International Conference on Computer Vision},
	year      = {2021},
}
```
                                                                                                                           
## License

This project is developed by Alibaba and licensed under the [Apache 2.0 license](LICENSE).

This product contains third-party components under other open source licenses.

See the [NOTICE](NOTICE) file for more information.
