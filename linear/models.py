from abc import abstractmethod
from typing import Dict
import math
import torch
from torch import nn
from torch.nn import init

class MyModelMixin:
    @abstractmethod
    def get_weights(self) -> Dict:
        pass

class Linear(nn.Linear, MyModelMixin):
    def get_weights(self):
        if self.bias:
            return {
                'weight': self.weight,
                'bias': self.bias
            }
        else:
            return {'weight': self.weight}


class SpaRedLinear(nn.Module, MyModelMixin):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpaRedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.weight2 = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, **factory_kwargs))
            self.bias2 = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.bias2, -bound, bound)

    def forward(self, X, **kwargs):
        weight = self.weight * self.weight2
        if self.use_bias:
            bias = self.bias * self.bias2
            x = nn.functional.linear(X, weight, bias)
        else:
            x = nn.functional.linear(X, weight)
        assert not torch.isnan(x).any()
        return x

    def get_weights(self):
        if self.use_bias:
            return {'weight': self.weight * self.weight2,
                    'bias': self.bias * self.bias2}
        else:
            return {'weight': self.weight * self.weight2}

class SparedLinearRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=False):
        super(SparedLinearRegression, self).__init__()
        torch.manual_seed(111)
        self.linear = SpaRedLinear(input_dim, output_dim, bias)

    def forward(self, x):
        return self.linear(x)

    def get_weights(self):
        return self.linear.get_weights()
