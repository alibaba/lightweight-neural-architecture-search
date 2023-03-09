# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from abc import ABC, abstractmethod

class BudgetBase(ABC, ):

    def __init__(self,
                 name,
                 budget):
        super().__init__()
        self.budget_name = name
        self.budget_value = budget

    @abstractmethod
    def __call__(self, input):
        pass 

    @property
    def name(self,):
        return self.budget_name

    @property
    def budget(self,):
        return self.budget_value


        
