# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.
import copy
from numpy import random
from abc import ABC, abstractmethod
from .space_utils import (adjust_structures_inplace, __check_block_structure_info_list_valid__)

class BaseSpace(ABC):
    def __init__(self, name = None):
        self.name = name

    @abstractmethod
    def mutate(self, *args, **kwargs):
        return None 

    def choice(self,):
        return None

    def __call__(self, *args, **kwargs):
        return self.mutate(*args, **kwargs)

class CnnBaseSpace(BaseSpace):
    def __init__(self, name = None, image_size = 224, block_num = 2, exclude_stem = False, budget_layers=None, **kwargs): 

        super().__init__(name)
        self.budget_layers = budget_layers
        self.block_num = block_num
        self.exclude_stem = exclude_stem 
        self.image_size = image_size
        self.mutators = {}

    def mutate(self, block_structure_info_list,
                                  minor_mutation=False, *args, **kwargs):
        block_structure_info_list = copy.deepcopy(block_structure_info_list)

        for mutate_count in range(self.block_num):
            is_valid = False
            new_block_structure_info_list = block_structure_info_list

            for idx in range(len(block_structure_info_list)):
                random_id = random.randint(0, len(block_structure_info_list) - 1)

                if self.exclude_stem:
                    while random_id == 0:
                        random_id = random.randint(
                            0,
                            len(block_structure_info_list) - 1)

                mutated_block = self.mutators[block_structure_info_list[random_id] ['class']](
                            random_id,
                            block_structure_info_list,
                            minor_mutation=minor_mutation)

                if mutated_block is None:
                    continue

                new_block_structure_info_list = copy.deepcopy(block_structure_info_list)
                new_block_structure_info_list[random_id] = mutated_block

                adjust_structures_inplace(new_block_structure_info_list, self.image_size)
                if __check_block_structure_info_list_valid__(new_block_structure_info_list, self.budget_layers):
                    break
            pass  # end while not is_valid:
            block_structure_info_list = new_block_structure_info_list

        return block_structure_info_list
