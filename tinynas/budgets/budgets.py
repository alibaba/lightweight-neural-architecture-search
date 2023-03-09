from .builder import BUDGETS
from .base import BudgetBase

@BUDGETS.register_module(module_name = 'layers')
@BUDGETS.register_module(module_name = 'model_size')
@BUDGETS.register_module(module_name = 'flops')
@BUDGETS.register_module(module_name = 'latency')
@BUDGETS.register_module(module_name = 'max_feature')
@BUDGETS.register_module(module_name = 'efficient_score')
class ExceedBudget(BudgetBase):
    def __init__(self, name, budget, logger, **kwargs):
        super().__init__(name, budget)
        self.logger = logger 

    def compare(self, model_info):
        input  = model_info[self.name]
        if self.budget < input:
            self.logger.debug(
                '{} value = {} in the structure_info exceed the budget ={}'.
                format(self.name, input, self.budget))
            return False
        return True

    def __call__(self, model_info):
        return self.compare(model_info)
