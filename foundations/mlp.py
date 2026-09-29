import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        z = x
        n = len(weights) - 1
        for i, (W, b) in enumerate(zip(weights, biases)):
            z = z @ W + b
            if i != n:
                z *= (z>0).astype(float)
        return np.round(z, 5)

