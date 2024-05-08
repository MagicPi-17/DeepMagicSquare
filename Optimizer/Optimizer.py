import torch
from torch import optim
from LossFunctions.LossFunciton import LossFunction


class Optimizer:
    def __init__(self, params):
        self.loss_functions = []
        self.params = torch.tensor(params, requires_grad=True)
        self.square_length = int(len(params) ** 0.5)

    def add_loss_function(self, loss_function):
        if not issubclass(type(loss_function), LossFunction):
            raise ValueError("Loss function must be a subclass of LossFunction")
        self.loss_functions.append(loss_function)

    def total_loss(self):
        # Replace this with your custom combination of losses
        return sum([f.compute(self.params) for f in self.loss_functions])

    def minimize(self, learning_rate=0.01, num_iterations=100,
                 display_every=100, display_log=True):

        optimizer = optim.Adam([self.params], lr=learning_rate)
        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = self.total_loss()
            loss.backward()
            optimizer.step()

            if display_every is not None and display_log and i % display_every == display_every - 1:
                if display_log:
                    print(f"Iteration {i + 1}: Parameters =")
                    rounded_params = self.params.round()
                    rounded_params = rounded_params.reshape(self.square_length, self.square_length)
                    row_sums = rounded_params.sum(dim=1)

                    # Calculate the sum of each column
                    col_sums = rounded_params.sum(dim=0)

                    # Calculate the sum of the diagonals
                    diag1_sum = rounded_params.diag().sum()
                    diag2_sum = torch.flip(rounded_params, [0]).diag().sum()


                    for idx, row in enumerate(rounded_params):
                        print(f"{row.tolist()} {row_sums[idx].item()}")

                    # Print the sum of each column
                    print(f"\n{col_sums.tolist()}")
                    print(f"Sum of diagonals: {[diag1_sum.item(), diag2_sum.item()]}\n")


                    for k, f in enumerate(self.loss_functions):
                        raw_loss = f.raw_loss(self.params).detach().numpy()
                        print(f"Iteration {i + 1}, {type(f).__name__} Raw Loss = {raw_loss}", f.reverse_momentum)
                    print(f"Iteration {i + 1}: Scaled Total Loss = {loss.detach().numpy()}")

                    print("=" * 20)
        return self.params
