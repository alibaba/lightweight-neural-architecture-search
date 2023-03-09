## Abstract

* **Instruction**

    In this folder, we provide the structure txt and parameters of the model searched by TinyNAS.  <br/><br/>

* **Use the searching configs for Classification**
    ```shell
    sh tools/dist_search.sh configs/MBV2_FLOPs.py
    ```
    **`MBV2_FLOPs.py` is the config for searching MBV2-like model within the budget of FLOPs.**

    **`R50_FLOPs.py` is the config for searching R50-like model within the budget of FLOPs.** 
      
    **`deepmad_R18_FLOPs.py` is the config for searching R18-like model within the budget of FLOPs using DeepMAD.**  

    **`deepmad_R34_FLOPs.py` is the config for searching R34-like model within the budget of FLOPs using DeepMAD.**  

    **`deepmad_R50_FLOPs.py` is the config for searching R50-like model within the budget of FLOPs using DeepMAD.**  
    
    **`deepmad_29M_224.py` is the config for searching 29M SoTA model with 224 resolution within the budget of FLOPs using DeepMAD.**  
    
    **`deepmad_29M_288.py` is the config for searching 29M SoTA model with 288 resolution within the budget of FLOPs using DeepMAD.**  
    
    **`deepmad_50M.py` is the config for searching 50M SoTA model within the budget of FLOPs using DeepMAD.**  

    **`deepmad_89M.py` is the config for searching 89M SoTA model within the budget of FLOPs using DeepMAD.**  <br/><br/>


* **Use searched models in your own training pipeline**

    **copy `tinynas/deploy/cnnnet` to your pipeline, then** 
    ```python
    from cnnnet import CnnNet
    # for classifictaion
    model = CnnNet(num_classes=classes, 
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
| R18-like | 224 | 10.8 |    1.7     |   78.44  | [txt](models/R18-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/TinyNAS/classfication/R18-like.pth.tar) |
| R50-like | 224 | 21.3 |    3.6     |   80.04  | [txt](models/R50-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/TinyNAS/classfication/R50-like.pth.tar) |
| R152-like | 224 | 53.5 |    10.5     |   81.59  | [txt](models/R152-like.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/TinyNAS/classfication/R152-like.pth.tar) |


**Note**:

1. These models are trained on ImageNet dataset with 8 NVIDIA V100 GPUs.
2. Use SGD optimizer with momentum 0.9; weight decay 5e-5 for ImageNet; initial learning rate 0.1 with 480 epochs.

***
## Citation

If you find this toolbox useful, please support us by citing this work as
```
@inproceedings{cvpr2023deepmad,
	title      = {DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network},
	author     = {Xuan Shen, Yaohua Wang, Ming Lin, Dylan Huang, Hao Tang, Xiuyu Sun, Yanzhi Wang},
	booktitle  = {Conference on Computer Vision and Pattern Recognition 2023},
	year       = {2023},
	url        = {https://arxiv.org/abs/2303.02165}
}
```

```
@inproceedings{zennas,
	title     = {Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition},
	author    = {Ming Lin and Pichao Wang and Zhenhong Sun and Hesen Chen and Xiuyu Sun and Qi Qian and Hao Li and Rong Jin},
	booktitle = {2021 IEEE/CVF International Conference on Computer Vision},  
	year      = {2021},
}
```
