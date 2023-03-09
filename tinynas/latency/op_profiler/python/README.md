***
### example
1. generate the sampling txt with the  configurations.
    ```shell
    python sample.py --config path_to_config_file --nbits 8 --save-file sample
    ```

2. Use the shell to inference each convolution with specific program on device, generate log.
    ```shell
    sh sample.sh
    ```

3. read the log to generate the final lib.

    ```shell
    python read_log.py
    ```
