# Commence usage

For initial usage, please follow the steps below sequentially.

## Search for the model architecture.

- Directories [configs/classification](configs/classification) and [configs/detection](configs/detection) respectively contain search configuration files for classification and detection, as demonstrated below.
- Taking a classification task as an example, search for the CNN architecture of ResNet-50 and execute the following steps:

    ```shell
    sh tools/dist_search.sh configs/classification/R50_FLOPs.py
    #or 
    python tools/search.py configs/classification/R50_FLOPs.py
    ```

## Export the search results.

- [tools/export.py](tools/export.py), A script for exporting the searched model architecture and its related dependencies is provided, allowing for quick verification of the demo

- For example [R50_FLOPs](configs/classification/R50_FLOPs.py)：

    ```shell
    python tools/export.py save_model/R50_R224_FLOPs41e8 output_dir
    ```

    Copy the demo deployment and related code to the **output_dir/R50_R224_FLOPs41e8/** directory, which should include the following contents:

    - best_structure.json：Several optimal model architectures that were found during the search.
    - demo.py：A simple script demonstrating how to use the models
    - cnnnet.py：The class definitions and utility functions used for constructing the models
    - modules： The foundational modules of the models.
    - weights/：Several optimal model weights that were found during the search (only available for one-shot NAS methods).


## Using the searched architecture.

- [demo.py](tinynas/deploy/cnnnet/demo.py) is a basic usage example, but you can also run demo.py directly after exporting the model architecture in the previous step.

- Continuing with the ResNet-50 architecture for a classification task as an example, the core code is explained below:

    - Import dependencies

    ```python
    import ast
    from cnnnet import CnnNet
    ```

    - Load the optimal structure from a file."

    ```python
    with open('best_structure.json', 'r') as fin:
        content = fin.read()
        output_structures = ast.literal_eval(content)

    network_arch = output_structures['space_arch']
    best_structures = output_structures['best_structures']
    ```

    - Instantiate the classification backbone network.

    ```python
    network_id = 0    # Index number. Multiple structures can be output during the search, as set by num_network=5 in example_cls_res50.sh.
    out_indices = (4, )    # Output stage. For classification tasks, only the output from the final stage needs to be obtained.

    backbone = CnnNet(
        structure_info=best_structures[network_id],
        out_indices=out_indices,
        num_classes=1000,
        classification=True,
        )
    backbone.init_weight(pretrained)
    ```

    - You can now fully utilize the `backbone` :smile:

- For further usage methods of the CNN detection task model, please refer to [tinynas/deploy/](tinynas/deploy/)
