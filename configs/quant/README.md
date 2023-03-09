## Abstract

* **Instruction**

    We propose Quantization Entropy Score (QE-Score) to calculate the entropy for searching efficient low-precision backbones. In this folder, we provide the example scripts and structure txt for quantization models, which are aligned with MobileNetV2-4/8bit. 
    Mixed7d0G is aligned with MobileNetV2-4bit, while Mixed19d2G is aligned with MobileNetV2-8bit. The training pipeline is released on the [QE-Score official repository](https://github.com/implus/GFocalV2).  <br/><br/>


* **Use the searching examples for Quantization**
    ```shell
    sh tools/dist_search.sh  Mixed_7d0G.py
    ```  
    **`mixed_7d0G.py` is the config for searching Mixed7d0G model within the budget of FLOPs of MobileNetV2-4bit.**

    **`mixed_19d2G.py` is the config for searching Mixed19d2G model within the budget of FLOPs of MobileNetV2-8bit.**

***

## Results and Models

|Backbone|Param (MB)|BitOps (G)|ImageNet TOP1|Structure|Download|
|:----|:----|:----|:----|:----|:----|
|MBV2-8bit|3.4|19.2|71.90%| -| -|
|MBV2-4bit|2.3|7|68.90%| -|- |
|Mixed19d2G|3.2|18.8|74.80%|[txt](models/mixed7d0G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-7d0G/quant_238_70.7660.pth.tar) |
|Mixed7d0G|2.2|6.9|70.80%|[txt](models/mixed19d2G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-19d2G/quant_237_74.8180.pth.tar) |

The ImageNet training pipeline can be found at [https://github.com/tinyvision/imagenet-training-pipeline ](https://github.com/tinyvision/imagenet-training-pipeline)

**Note**:
- If searching without quantization, Budget_flops is equal to the base flops as in other tasks.
- If searching with quantization, Budget_flops = Budget_flops_base x (Act_bit / 8bit) x (Weight_bit / 8bit). Hence, BitOps = Budget_flops x 8 x 8.


***
## Citation

If you use this toolbox in your research, please cite the paper.

```
@inproceedings{qescore,
	title     = {Entropy-Driven Mixed-Precision Quantization for Deep Network Design},
	author    = {Zhenhong Sun and Ce Ge and Junyan Wang and Ming Lin and Hesen Chen and Hao Li and Xiuyu Sun},
	journal   = {Advances in Neural Information Processing Systems},
	year      = {2022},
}
