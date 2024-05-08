from .LossFunciton import LossFunction
import torch

class SumLoss(LossFunction):
    def __init__(self, importance, n):
        super().__init__(importance)
        self.total_sum = (n + 1) * n / 2

    def raw_loss(self, params):
        return ((torch.sum(torch.abs(params)) - self.total_sum) ** 2) / self.total_sum
