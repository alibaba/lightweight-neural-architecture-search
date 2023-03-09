# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import logging


def get_logger(name,
               rank=0,
               log_file=None,
               log_level="INFO",
               file_mode='w'):
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
    if log_level.upper() == 'DEBUG':
        log_level = logging.DEBUG
    elif log_level.upper() == 'ERROR':
        log_level = logging.ERROR
    elif log_level.upper() == 'WARNING':
        log_level = logging.WARNING
    else: 
        log_level = logging.INFO

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
        f'%(asctime)s-%(name)s-%(levelname)s-rank{rank}: %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger

def get_root_logger(name='Search',
                    rank=0,
                    log_file=None,
                    log_level=logging.INFO):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to 'nas'.
        log_file ([type], optional): [description]. Defaults to None.
        log_level ([type], optional): [description]. Defaults to logging.INFO.

    Returns:
        [type]: [description]
    """
    logger = get_logger(
        name=name, rank=rank, log_file=log_file, log_level=log_level)

    return logger


class MyLogger():

    def __init__(self, log_filename=None, verbose=False):
        self.log_filename = log_filename
        self.verbose = verbose
        if self.log_filename is not None:
            mkfilepath(self.log_filename)
            self.fid = open(self.log_filename, 'w')
        else:
            self.fid = None

    def info(self, msg):
        msg = str(msg)
        print(msg)
        if self.fid is not None:
            self.fid.write(msg + '\n')
            self.fid.flush()

    def debug_info(self, msg):
        if not self.verbose:
            return
        msg = str(msg)
        print(msg)
        if self.fid is not None:
            self.fid.write(msg + '\n')
            self.fid.flush()
