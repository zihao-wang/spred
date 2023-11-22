import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class SpredLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpredLinear, self).__init__()
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


h = nn.Conv2d
class SpredConv(nn.Module):
    ## TORCH.NN.FUNCTIONAL.CONV2D

    def __init__(self, in_kernel: int, out_kernel: int, kernel_size=1, stride=1, padding=0, bias=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpredConv, self).__init__()

        conv2 = h(in_kernel, out_kernel, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)

        self.in_features = in_kernel
        self.out_features = out_kernel

        self.weight = nn.Parameter(torch.empty(
            conv2.weight.data.shape, **factory_kwargs))
        self.weight2 = nn.Parameter(torch.empty(
            conv2.weight.data.shape, **factory_kwargs))
        self.use_bias = bias

        self.stride = stride
        self.padding = padding
        #if bias:
        #    self.bias = nn.Parameter(torch.empty(
        #        out_features, **factory_kwargs))
        #    self.bias2 = nn.Parameter(
        #        torch.empty(out_features, **factory_kwargs))
        #else:
        #    self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        #if self.bias is not None:
        ##    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #    init.uniform_(self.bias, -bound, bound)
        #    init.uniform_(self.bias2, -bound, bound)

    def forward(self, X, **kwargs):
        weight = self.weight * self.weight2

        #weight = self.weight * self.weight2

        x = F.conv2d(X, weight, stride=self.stride, padding=self.padding)
        assert not torch.isnan(x).any()
        return x
