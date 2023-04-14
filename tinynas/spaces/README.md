## Search Space module
***
**For NAS, search space is very important. Referring to previous network design experience, designing the search space you want can ensure effective search results.**
***

### **Supported Search Space**
* `space_k1kxk1.py`: Base Resnet-like search space for Classification and Detection.
* `space_k1dwk1.py`: Base MobileNetV2-like search space.
* `space_k1dwsek1.py`: Base MobileNetV2-se block search space.
* `space_k1kx.py`: Base small and tiny DAMO-YOLO search space.
* `space_kxkx.py`: Base medium DAMO-YOLO search space.
* `space_quant_k1dwk1.py`: Base search space for Quantization search.
* `space_3d_k1dwk1.py`: Base search space for 3D CNN search.
