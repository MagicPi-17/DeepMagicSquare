import torch
import numpy as np

def getEquations(n):
    equations = []

    row_equation = []
    column_equation = []

    for i in range(n ** 2):
        row_equation.append(i)
        column_equation.append((i % n) * n + i // n)
        if i % n == n - 1:
            equations.append(row_equation)
            equations.append(column_equation)
            row_equation = []
            column_equation = []

    dig = [[], []]
    for i in range(n):
        dig[0].append(i * (n + 1))
        dig[1].append((i + 1) * (n - 1))

    equations += dig
    return torch.tensor(equations)


def sigmoid(x, range):
    return 1/(1 + np.exp(-x*range))  * range

def tanh(x, range):
    return np.tanh(x / range)  * range
