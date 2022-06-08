## Abstract

* **Instruction**

    In this folder, we provide the structure txt and parameters of the model searched by LightNAS.  <br/><br/>

* **Use the searching examples for Classification**
    ```shell
    sh example_MBV2_FLOPs.sh or example_R50_FLOPs.sh
    ```
    **`example_MBV2_FLOPs.sh` is the script for searching MBV2-like model within the budget of FLOPs.**

    **`example_R50_FLOPs.sh` is the script for searching R50-like model within the budget of FLOPs.**

    **`example_entropy.sh` is the script for debugging the computation of the entropy score, which only show the masternet information.**  <br/><br/>


* **Use searched models in your own training pipeline**

    **copy `nas/models` to your pipeline, then** 
    ```python
    from models import __all_masternet__
    # for classifictaion
    model = __all_masternet__['MasterNet'](num_classes=classes, 
                                structure_txt=structure_txt,
                                out_indices=(4,),
                                classfication=True)

    # if load with pretrained model
    model.init_weights(pretrained=pretrained_pth)
    ```
***

## Results and Models

| Backbone  | size   | Param (M) | FLOPs (G) |   Top-1 | Structure | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:------:|
| R18-like | 224 | 10.8 |    1.7     |   78.44  | [txt](models/R18-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R18-like.pth.tar) |
| R50-like | 224 | 21.3 |    3.6     |   80.04  | [txt](models/R50-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R50-like.pth.tar) |
| R152-like | 224 | 53.5 |    10.5     |   81.59  | [txt](models/R152-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/classfication/R152-like.pth.tar) |


**Note**:

1. These models are trained on ImageNet dataset with 8 NVIDIA V100 GPUs.
2. Use SGD optimizer with momentum 0.9; weight decay 5e-5 for ImageNet; initial learning rate 0.1 with 480 epochs.

***
## Citation

If you find this toolbox useful, please support us by citing this work as

```
@inproceedings{zennas,
	title     = {Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition},
	author    = {Ming Lin, Pichao Wang, Zhenhong Sun, Hesen Chen, Xiuyu Sun, Qi Qian, Hao Li and Rong Jin},
	booktitle = {2021 IEEE/CVF International Conference on Computer Vision},  
	year      = {2021},
}
```