## Models Definition module

The basic network (supernet) is defined by the unified framework with the structure info list. CnnNet is the overall backbone network, which is composed of differet downsample stages with structure info list. A stage is composed of several blocks, such as Resnet bottleneck block, MobileV2 Depthwise block. Basic block is composed of 2D convolutions.

***
### **SuperModel**
* **`cnnnet.py`**: Define the backbone network for CnnNet, such as classification and detection.

***
### **Typical Structure Info List for CnnNet**
```
[{'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 2, 'k': 3},
{'class': 'SuperResK1KXK1', 'in': 32, 'out': 256, 's': 2, 'k': 3, 'L': 1, 'btn': 64},
{'class': 'SuperResK1KXK1', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 1, 'btn': 128},
{'class': 'SuperResK1KXK1', 'in': 512, 'out': 1024, 's': 2, 'k': 3, 'L': 1, 'btn': 256},
{'class': 'SuperResK1KXK1', 'in': 1024, 'out': 2048, 's': 2, 'k': 3, 'L': 1, 'btn': 512}, ]
```
***
### **Supported Blocks**
`Supported 2D CNN blocks:`
```
__all_blocks__ = {
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
    'ResK1KXK1': ResK1KXK1,
    'ResK1KX': ResK1KX,
    'ResKXKX': ResKXKX,
    'ResK1DWK1': ResK1DWK1,
    'ResK1DWSEK1': ResK1DWSEK1,
    'SuperResK1KXK1': SuperResK1KXK1,
    'SuperResK1KX': SuperResK1KX,
    'SuperResKXKX': SuperResKXKX,
    'SuperResK1DWK1': SuperResK1DWK1,
    'SuperResK1DWSEK1': SuperResK1DWSEK1,
    'SuperQuantResK1DWK1': SuperQuantResK1DWK1,
} 
```
`Supported 3D CNN blocks:`

```
__all_blocks_3D__ = {
    'Conv3DKXBN': Conv3DKXBN,
    'Conv3DKXBNRELU': Conv3DKXBNRELU,
    'BaseSuperBlock3D': BaseSuperBlock3D,
    'Res3DK1DWK1': Res3DK1DWK1,
    'SuperRes3DK1DWK1': SuperRes3DK1DWK1,
}
```
**Note**:

- `BaseSuperBlock` is the basic class for super block.
- `SuperResK1KXK1` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResK1KXK1`.
- `SuperResK1DWK1` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResK1DWK1`.
- `SuperResK1DWSEK1` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResK1DWSEK1`.
- `SuperResK1KX` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResK1KX`.
- `SuperResKXKX` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResKXKX`.
- `SuperQuantResK1DWK1` is the derived class from ``SuperResK1DWK1``.
- `SuperRes3DK1DWK1` is the derived class from ``BaseSuperBlock`` to unit `L` class `Res3DK1DWK1`.
***
### **Useful functions for masternet**

`get_model_size`: Get the number of parameters of the network

`get_flops`: Get the FLOPs of the network

`get_layers`: Get the Conv layers of the network

`get_latency`: Get the latency from predictor or benchmark

`get_params_for_trt`: Get the paramters of the network for latency prediction.

`get_max_feature`: Get the number of max feature map for MCU.

`get_efficient_score`: Get efficient_score of the network.

`madnas_forward_pre_GAP`: Get the madnas score of the network, which does not need forward on GPU and has very fast speed .

`deepmad_forward_pre_GAP`: Get the deepmad score of the network, which does not need forward on GPU and has very fast speed. 

`stentr_forward_pre_GAP`: Get the spatio-temporal entropy score of the network, which does not need forward on GPU and has very fast speed.

***
