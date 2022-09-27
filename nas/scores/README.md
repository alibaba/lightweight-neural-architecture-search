## Score module

***

### **Entropy Score**

`For Entropy Score, please refer to this. `[Example Shell](/scripts/classification/example_entropy.sh)
```
@inproceedings{maedet,
  title={MAE-DET: Revisiting Maximum Entropy Principle in Zero-Shot NAS for Efficient Object Detection},
  author={Zhenhong Sun, Ming Lin, Xiuyu Sun, Zhiyu Tan, Hao Li and Rong Jin},
  booktitle={International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}
```

***

### **MadNAS Score**: 
The version of mathematical formula calculation, which does not need forward on GPU and runs very fast compared with Entropy Score. [Example Shell](/scripts/classification/example_madnas.sh)

`By incorporating quantization, we further provide QE-Score.`
```
@article{qescore,
  title     = {Entropy-Driven Mixed-Precision Quantization for Deep Network Design on IoT Devices},
  author    = {Zhenhong Sun and Ce Ge and Junyan Wang and Ming Lin and Hesen Chen and Hao Li and Xiuyu Sun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
}
```
