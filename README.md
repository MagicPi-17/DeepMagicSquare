# DeepMagicSquare

DeepMagicSquare is a project that uses deep learning techniques to find solutions for magic squares. It optimizes a set of penalty functions to guide the solution towards a valid magic square.

## Penalty Functions

The optimization process is guided by three penalty functions:

### Sum Penalty

This function calculates the deviation of the sum of the parameters from a target sum. The penalty is the square of this deviation divided by the target sum.

```math
$$\text{{Sum Penalty}} = \frac{{(\sum |x_i| - \text{{total\_sum}})^2}}{{\text{{total\_sum}}}}$$
```
where $x_i$ represents the $i$-th element of the parameters.

### Gap Penalty

This function calculates the gaps between sorted parameters. The penalty is the sum of the squares of these gaps.

First, sort the parameters:

```math
$$\text{{sorted\_params}} = \text{{sort}}(x)$$
```

Then, calculate the gaps and the penalty:

```math
$$\text{{gaps}} = (1 - | \text{{sorted\_params}}[i+1] - \text{{sorted\_params}}[i] |)^2$$
```

```math
\text{{Gap Penalty}} = \sum \text{{gaps}}
```

### Equality Penalty

This function calculates the deviation of the sum of each row, each column, and each diagonal in the magic square from a target value. The penalty is the sum of these deviations. 

For a 3x3 magic square with a target sum of 15, each row, column, and diagonal should sum to 15. The penalty function is calculated as follows:

First, calculate the sum of the elements in each row, column, and diagonal:

```math
$$\text{{elements\_sum}} = \sum x[\text{{rows\_elements}}], \sum x[\text{{columns\_elements}}], \sum x[\text{{diagonals\_elements}}]$$
```
Then, calculate the penalty:

```math
$$\text{{Equality Penalty}} = \sum | \text{{elements\_sum}} - 15 |$$
```

## Optimization Technique

This project uses a minor reverse momentum technique for optimization. If the correct solution is not found, the learning rate increases to expedite the search for the solution.

## License

This project is licensed under the terms of the MIT license.
