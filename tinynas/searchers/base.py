# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from abc import ABC, abstractmethod
from modelscope.utils.config import Config

class BaseSearcher(ABC):
    def __init__(self, cfg_file: str):
        """ Searcher basic init, should be called in derived class

        Args:
            cfg_file: Path to configuration file.
            arg_parse_fn: Same as ``parse_fn`` in :obj:`Config.to_args`.
        """
        self.cfg = Config.from_file(cfg_file)

    @abstractmethod
    def run(self, *args, **kwargs):
        """ search process

        Train process should be implemented for specific task or
        model, releated paramters have been intialized in
        ``BaseSearcher.__init__`` and should be used in this function
        """
        pass
