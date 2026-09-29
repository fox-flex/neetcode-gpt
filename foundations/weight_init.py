import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std

        return torch.round(W, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in))
        W = torch.randn(fan_out, fan_in) * std

        return torch.round(W, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        
        
        Ws = []
        for _ in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2 / (input_dim + hidden_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2 / input_dim)
            elif init_type == 'random':
                std = 1.0
            else:
                raise 1
            
            W = torch.randn(hidden_dim, input_dim) * std
            Ws.append(W)
        
        z = torch.randn(input_dim)
        res = []
        for W in Ws:
            z = torch.relu(W @ z)
            res.append(round(z.std().item(), 2))
            input_dim = hidden_dim
        
        return res


