## Models Definition module

The basic network defined by the unified framework with the structure info list.
- Masternet is the overall backbone network, which is composed of differet downsample stages with structure info list. 
- A stage is composed of several blocks, such as Resnet bottleneck block and MobileV2 Depthwise block.
- Basic block is composed of 2D convolutions. 

***
### **Masternet**
- **`masternet.py`**: Define the backbone network for classification and detection.
- **`masterxxx.py`**: You can define your own backbone network for other tasks.

***
### **Typical Structure Info List**
```
[{'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 2, 'k': 3}, 
{'class': 'SuperResConvK1KXK1', 'in': 32, 'out': 256, 's': 2, 'k': 3, 'L': 1, 'btn': 64}, 
{'class': 'SuperResConvK1KXK1', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 1, 'btn': 128}, 
{'class': 'SuperResConvK1KXK1', 'in': 512, 'out': 1024, 's': 2, 'k': 3, 'L': 1, 'btn': 256}, 
{'class': 'SuperResConvK1KXK1', 'in': 1024, 'out': 2048, 's': 2, 'k': 3, 'L': 1, 'btn': 512}, ]
```
***
### **Supported Blocks**
`Supported 2D blocks:`
```
__all_blocks__ = {
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
    'ResConvK1KXK1': ResConvK1KXK1,
    'SuperResConvK1KXK1': SuperResConvK1KXK1,
    'ResK1DWK1': ResK1DWK1,
    'SuperResK1DWK1': SuperResK1DWK1,
}
```
**Note**:

- `BaseSuperBlock` is the basic class for super block.
- `SuperResConvK1KXK1` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResConvK1KXK1`.
- `SuperResK1DWK1` is the derived class from ``BaseSuperBlock`` to unit `L` class `ResK1DWK1`.

***
### **Useful functions for masternet**

`get_model_size`: Get the number of parameters of the network

`get_flops`: Get the FLOPs of the network

`get_num_layers`: Get the Conv layers of the network

`get_num_stages`: Get the number of downsample stages of the network

`get_params_for_trt`: Get the paramters of the network for latency prediction.

`entropy_forward_pre_GAP`: Get the entropy score of the network, which need forward on GPU.

`madnas_forward_pre_GAP`: Get the madnas score of the network, which does not need forward on GPU and runs very fast compared with entropy score.

`get_max_feature_num`: Get the number of max feature map for MCU.
