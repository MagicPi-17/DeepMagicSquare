from .LossFunciton import LossFunction
import torch


class GapLoss(LossFunction):
    def __init__(self, importance, n):
        super().__init__(importance)
        self.n = n - 1

    def raw_loss(self, params):
        sorted_params, _ = torch.sort(params)
        gaps = torch.square(1 - torch.abs(sorted_params[1:] - sorted_params[:-1]))
        return torch.sum(gaps) / self.n
