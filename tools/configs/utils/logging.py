# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import logging


def get_logger(name, rank=0, log_file=None, log_level=logging.INFO, file_mode='w'):
    """[summary]

    Args:
        name ([type]): [description]
        rank (int, optional): [description]. Defaults to 0.
        log_file ([type], optional): [description]. Defaults to None.
        log_level ([type], optional): [description]. Defaults to logging.INFO.
        file_mode (str, optional): [description]. Defaults to 'w'.

    Returns:
        [type]: [description]
    """
    logger = logging.getLogger(name)

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    # all rank will add a StreamHandler for error output
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s-%(name)s-%(levelname)s: %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger