from itertools import chain
from turtle import forward
import torch
from torch import nn as _nn
import torch.nn.functional as F
from .sparse_modules import SpredConv, SpredLinear

_nn.Conv2d = SpredConv
_nn.Linear = SpredLinear

class MLP(_nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_classes) -> None:
        super(MLP, self).__init__()
        self.input = SpredLinear(input_dim, hidden_dim)
        self.mlp = _nn.Sequential(
            *chain([
                [SpredLinear(hidden_dim, hidden_dim), _nn.ReLU()]
                for _ in range(num_layers-1)
            ])
        )
        self.cls = SpredLinear(hidden_dim, num_classes)

    def forward(self, X):
        x = torch.relu(self.input(X))
        x = self.mlp(x)
        return self.cls(x)
