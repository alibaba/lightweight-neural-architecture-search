## Abstract

* **Instruction**

    We search efficient E3D backbones for action recognition and E3D-S/M/L are aligned with X3D-S/M/L.   <br/>


* **Use the searching configs for Classification**

    ```shell
    sh tools/dist_search.sh configs/E3D_X3DS_FLOPs.py
    ```
    **`E3D_X3DS_FLOPs.py` is the config for searching X3DS-like model within the budget of FLOPs using STEntr Score.**

    **`E3D_X3DM_FLOPs.py` is the config for searching X3DM-like model within the budget of FLOPs using STEntr Score.** 
      
    **`E3D_X3DL_FLOPs.py` is the config for searching X3DL-like model within the budget of FLOPs using STEntr Score.**

* **Use searched models in your own training pipeline**

    **copy `tinynas/deploy/cnn3dnet` to your pipeline, then** 
    ```python
    from cnn3dnet import Cnn3DNet
    # for classifictaion
    model = Cnn3DNet(num_classes=classes, 
                    structure_txt=structure_txt,
                    out_indices=(4,),
                    classfication=True)

    # if load with pretrained model
    model.init_weights(pretrained=pretrained_pth)
    ```

***

## Results on Sth-Sth V1

| Backbone  | size   |  FLOPs (G) |   Top-1 | Top-5 | Structure | 
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|
| E3D-S | 160 |    1.9     |   47.1  | 75.6| [txt](models/E3D_S.txt)       |
| E3D-M  | 224 |     4.7     |   49.4  | 78.1| [txt](models/E3D_M.txt)       |
| E3D-L  | 312 |     18.3     |   51.1  | 78.7| [txt](models/E3D_L.txt)       |



***
## Citation

If you find this toolbox useful, please support us by citing this work as

```
@inproceedings{iclr23maxste,
	title     = {Maximizing Spatio-Temporal Entropy of Deep 3D CNNs for Efficient Video Recognition},
	author    = {Junyan Wang and Zhenhong Sun and Yichen Qian and Dong Gong and Xiuyu Sun and Ming Lin and Maurice Pagnucco and Yang Song },
	journal   = {International Conference on Learning Representations},
	year      = {2023},
}
```
