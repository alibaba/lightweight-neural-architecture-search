# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import random
from .mutator.base import Sampler

class RandomSampler(Sampler):
    def choice(self, candidates):
        return random.choice(candidates)

