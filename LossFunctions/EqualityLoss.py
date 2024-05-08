from .LossFunciton import LossFunction
from .helperFunctions.helperFunctions import *
import torch
import numpy as np

class EqualityLoss(LossFunction):
    def __init__(self, importance, n, targetValue):
        super().__init__(importance)
        self.n = n - 1
        self.targetValue = targetValue
        self.equations_indices = getEquations(n)

    def raw_loss(self, params):
        # indices_sum = lambda tensor, indices: torch.sum(tensor[indices])
        # params_eq_sum = torch.sum(
        #     torch.stack(
        #         [torch.abs(indices_sum(params, indices) - self.targetValue) for indices in self.equations_indices]))
        p_params = torch.round(params)
        params = (p_params * 0.9 + params * 0.1)
        indices_sum = lambda tensor, indices: torch.sum(tensor[indices], dim=1)
        params_eq_sum = torch.sum(torch.abs(indices_sum(params, self.equations_indices) - self.targetValue))

        return params_eq_sum
