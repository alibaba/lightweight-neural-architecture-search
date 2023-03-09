# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from numpy import random
from .base  import Mutator
from .builder import MUTATORS
from ..space_utils import smart_round

@MUTATORS.register_module(module_name = 'ChannelMutator')
class ChannelMutator(Mutator):
    def __init__(self, candidates, the_maximum_channel = 2048, *args, **kwargs):
        super().__init__(candidates = candidates)
        self.the_maximum_channel = the_maximum_channel

    def mutate(self, channels, *args, **kwargs):
        scale = self.sample()
        new_channels = smart_round(scale * channels)
        new_channels = min(self.the_maximum_channel, new_channels)
        return new_channels 
       
@MUTATORS.register_module(module_name = 'KernelMutator')
class KernelMutator(Mutator):
    def __init__(self, candidates, *args, **kwargs):
        super().__init__(candidates = candidates)

    def mutate(self, kernel_size, *args, **kwargs):
        for i in range(self.length()):
            new_kernel_size = self.sample()
            if new_kernel_size != kernel_size :
                break
        return new_kernel_size 

@MUTATORS.register_module(module_name = 'LayerMutator')
class LayerMutator(Mutator):
    def __init__(self, candidates, *args, **kwargs):
        super().__init__(candidates = candidates)

    def mutate(self, layer, *args, **kwargs):
        for i in range(self.length()):
            new_layer = layer + self.sample()
            new_layer = max(1, new_layer)
            if new_layer != layer:
                break
        return new_layer

@MUTATORS.register_module(module_name = 'BtnMutator')
class BtnMutator(Mutator):
    def __init__(self, candidates, *args, **kwargs):
        super().__init__(candidates = candidates)

    def mutate(self, btn_ratio, *args, **kwargs): 
        for i in range(self.length()):
            new_btn_ratio = self.sample()
            if new_btn_ratio != btn_ratio:
                break
        return new_btn_ratio

@MUTATORS.register_module(module_name = 'NbitsMutator')
class NbitsMutator(Mutator):
    def __init__(self, candidates, *args, **kwargs):
        super().__init__(candidates = candidates)

    def mutate(self, nbits, *args, **kwargs): 
        # avoid the endless loop
        if self.length() == 1 and nbits in self.candidates:
            return nbits
        ind = self.candidates.index(nbits)
        new_ind = ind

        while new_ind == ind:
            new_ind = random.choice((ind - 1, ind + 1))
            new_ind = max(0, new_ind)
            new_ind = min(self.length() - 1, new_ind)
            return self.candidates[new_ind]

@MUTATORS.register_module(module_name = 'NbitsListMutator')
class NbitsListMutator(Mutator):
    def __init__(self, candidates, nbits_ratio,*args, **kwargs):
        super().__init__(candidates = candidates)
        self.mutator = NbitsMutator(candidates = candidates)
        self.nbits_ratio = nbits_ratio

    def mutate(self, nbits_list, L, *args, **kwargs): 
        if isinstance(nbits_list, int):
            return self.mutator(nbits_list)
        else:
            inner_layer = len(nbits_list)//L
            for layer_idx in range(L):
                if random.uniform(0, 1) > self.nbits_ratio:
                    nbits_list[layer_idx*inner_layer:(layer_idx+1)*inner_layer] = \
                    [self.mutator(nbits_list[layer_idx*inner_layer])]*inner_layer
            return nbits_list
