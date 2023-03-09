# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from typing import (Any, Iterable, List, Optional, Tuple, cast)
import random
from .builder import MUTATORS
Choice = Any

class Sampler:
    """
    Handles `Mutator.choice()` calls.
    """

    def choice(self, candidates: List[Choice], *args, **kwargs): 
        raise NotImplementedError()

    def mutation_start(self, *args, **kwargs): 
        pass

    def mutation_end(self, *args, **kwargs): 
        pass


class RandomSampler(Sampler):
    def choice(self, candidates, *args, **kwargs):
        return random.choice(candidates)

@MUTATORS.register_module(module_name = 'BaseMutator')
class Mutator:

    def __init__(self, sampler = None, candidates = None, *args, **kwargs): 
        self.sampler: Optional[Sampler] = sampler
        self.candidates = candidates 
        if self.sampler is None:
            self.sampler = RandomSampler()
        self._length = len(self.candidates)

    def length(self):
        return self._length

    def bind_sampler(self, sampler: Sampler) -> 'Mutator':
        """
        Set the sampler which will handle `Mutator.choice` calls.
        """
        self.sampler = sampler
        return self

    def __call__(self, *args, **kwargs) ->None:
        return self.mutate(*args, **kwargs)

    def mutate(self, *args, **kwargs) -> None:
        """
        Abstract method to be implemented by subclass.
        Mutate a model in place.
        """
        ret = self.sampler.choice(list(self.candidates, *args, **kwargs)) 
        return ret 

    def sample(self, *args, **kwargs) -> None:
        """
        Abstract method to be implemented by subclass.
        Mutate a model in place.
        """
        ret = self.sampler.choice(list(self.candidates, *args, **kwargs)) 
        return ret 


