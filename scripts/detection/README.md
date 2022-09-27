## Abstract

* **<font size=4>Instruction</font>**

    <font size=3>We search efficient MAE-DET backbones for object detection and align with ResNet-50/101. MAE-DET-S uses 60% less FLOPs than ResNet-50; MAE-DET-M is aligned with ResNet-50 with similar FLOPs and number of parameters as ResNet-50; MAE-DET-L is aligned with ResNet-101.   <br/>

    In this folder, we provide the structure txt and parameters of the model searched by LightNAS for object detection. We follows [GFLV2 official repository](https://github.com/implus/GFocalV2) to train our models, and MAE-DET models are inserted into the pipeline. The code modification is shown in the following steps.</font>  <br/><br/>


* **<font size=4>Use the searching examples for MAE-DET</font>**
    ```shell
    sh example_R50_predictor.sh or example_R50_FLOPs.sh
    ```  
    **`example_R50_predictor.sh` is the script for searching R50-like  model within the budget of FLOPs and FP16 latency predictor on V100.**

    **`example_R50_FLOPs.sh` is the script for searching R50-like  model within the budget of FLOPs.**  <br/><br/>

* **<font size=4>Use MAE-DET models in GFLV2 official repository</font>**

    **1. Copy `nas/models` to `mmdet/backbones/` in GFLV2** 
    ```shell
    cd your_lightnas_path && cp -r nas/models your_gflv2_path/mmdet/backbones/
    ```

    **2. Copy `maedet_*.txt` to `gfocal_maedet/`, then copy the folder to `configs/`  in GFLV2** 
    ```shell
    cd models && cp maedet_*.txt gfocal_maedet/ && cp -r gfocal_maedet your_gflv2_path/configs/
    ```
    **3. Add `madnas.py` in `your_gflv2_path/mmdet/backbones/`.** 
    ```shell
    cd models && cp madnas.py your_gflv2_path/mmdet/backbones/
    ```
    **4. Add the following code snippet in `__init__.py` of `your_gflv2_path/mmdet/backbones/` by importing the initialization of MadNAS as shown bellow.** 
    ```python
    from .madnas import MadNas # add this

    __all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net','HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet', 
    "MadNas" # add this
    ]
    ```
    **5. Add the following code snippet after model building in Line 153 of `your_gflv2_path/tools/train.py`** 
    ```python
    if cfg.model.backbone.type in ["MadNas"] and cfg.use_syncBN_torch:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(f'Model:\n{model}')
    ```
    **6. Finally, follow the instruction of GFLV2 to train the models. If you add the MAE-DET backbones for other pipelines, please refer to this process.** 
***

## Results and Models

| Backbone | Param (M) | FLOPs (G) |   box AP<sub>val</sub> |   box AP<sub>S</sub> |   box AP<sub>M</sub>  |   box AP<sub>L</sub> | Structure | Download |
|:---------:|:---------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:------:|
| ResNet-50 | 23.5 |    83.6    |  44.7 | 29.1 | 48.1 | 56.6  | - | - |
| ResNet-101| 42.4 |   159.5    |  46.3 | 29.9 | 50.1 | 58.7  | - | - |
| MAE-DET-S | 21.2 |    48.7    |  45.1 | 27.9 | 49.1 | 58.0  | [txt](models/maedet_s.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-s/latest.pth) |
| MAE-DET-M | 25.8 |    89.9    |  46.9 | 30.1 | 50.9 | 59.9  | [txt](models/maedet_m.txt)       |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-m/latest.pth) |
| MAE-DET-L | 43.9 |    152.9   |  47.8 | 30.3 | 51.9 | 61.1  | [txt](models/maedet_l.txt)      |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/detection/maedet-l/latest.pth) |


**Note**:

1. The reported numbers here are from the newest simpler code implementation, which may be slightly better than that in the original paper.
2. These models are trained on COCO dataset with 8 NVIDIA V100 GPUs, where 3X learning schedule from scratch is applied.
3. Multi-scale training is used with single-scale testing, and the detailed training settings are in the corresponding config files.

***
## Citation

If you use this toolbox in your research, please cite the paper.

```
@inproceedings{maedet,
  title     = {MAE-DET: Revisiting Maximum Entropy Principle in Zero-Shot NAS for Efficient Object Detection},
  author    = {Zhenhong Sun and Ming Lin and Xiuyu Sun and Zhiyu Tan and Hao Li and Rong Jin},
  booktitle = {International Conference on Machine Learning},
  year      = {2022},
}
```
