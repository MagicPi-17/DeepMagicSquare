from Optimizer.Optimizer import Optimizer
from LossFunctions.GapLoss import GapLoss
from LossFunctions.SumLoss import SumLoss
from LossFunctions.EqualityLoss import EqualityLoss
import numpy as np
import torch




if __name__ == '__main__':
    n = 3
    n_square = n ** 2
    params = [torch.tensor(np.random.uniform(low=4.5, high=5.5), requires_grad=True) for _ in range(n_square)]

    optimizer = Optimizer(params)

    gap_error = GapLoss(importance=3, n=n_square)
    optimizer.add_loss_function(gap_error)

    sum_error = SumLoss(importance=2, n=n_square)
    optimizer.add_loss_function(sum_error)

    equality_error = EqualityLoss(importance=3, targetValue=15, n=n)
    optimizer.add_loss_function(equality_error)

    optimizer.minimize(learning_rate=0.1, num_iterations=1000, display_every=100, display_log=True)
