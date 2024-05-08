import torch
from .helperFunctions.helperFunctions import *

class LossFunction:
    def __init__(self, importance=1, reverse_momentum=1, previous_loss=1):
        self.importance = importance
        self.reverse_momentum= reverse_momentum 
        self.previous_loss = previous_loss

    def compute(self, params):
        raw_loss = self.raw_loss(params)
        scaled_loss =  self.importance * raw_loss

        new_momentum = 1 / (abs(self.previous_loss - scaled_loss.detach().numpy()) + 0.001)
        self.reverse_momentum = sigmoid(new_momentum, 2)
        self.previous_loss = scaled_loss.detach().numpy()
        return scaled_loss * new_momentum


    def raw_loss(self, params):
        raise NotImplementedError("Subclasses should implement this!")
