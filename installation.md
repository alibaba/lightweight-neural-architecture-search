# Installation

## Additional Requirements 
- Linux
- GCC 7+
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+

## Dependency Installation

1. Create a conda virtual environment and activate

    ```shell
    conda create -n tinynas python=3.6 -y
    conda activate tinynas
    ```

2. Install OpenMPI and mpi4py

    a. Install using conda command(recommended)

    ```shell
    conda install -c conda-forge mpi4py=3.0.3 openmpi=4.0.4
    ```

    b. Build from source (More faster when Multi-Process)
    - From [Here](https://www.open-mpi.org/software/ompi/v4.0/) download openmpi source code

    ```shell
    tar -xzvf openmpi-4.0.1.tar.gz
    cd openmpi-4.0.1
    ./configure --prefix=$HOME/openmpi
    make && make install
    ```
    - add mpi to the system path

    ```shell
    export PATH=$HOME/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/openmpi/lib:$LD_LIBRARY_PATH
    ```
    - install mpi4py

    ```shell
    # conda
    conda install -c conda-forge mpi4py
    # or pip
    pip install mpi4py
    ```

3. Run the following commands or  [Official Guide](https://pytorch.org/get-started/locally/) install torch and torchvision

    ```shell
    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
    ```
    > If you see `"Not be found for jpeg"`, please install libjpeg first.
    ```shell
    sudo yum install libjpeg # for centos
    sudo apt install libjpeg-dev # for ubuntu
    ```

4. ModelScope (optional)

    > For light-weight, we modify and port some code of modelscope , if you want to experience more features, you can visite [ModelScope](https://modelscope.cn/home)
    , or run the following installation commands
    ```shell
    pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
    ```
   
5. Other Requirements

    ```shell
    pip install -r requirements/nas.txt
    ```

6. Installation check
    ```python
    import tinynas 
    print(tinynas.__version__)
    ```
