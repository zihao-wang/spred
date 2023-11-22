from abc import abstractmethod
from typing import Dict
import math
import torch
from torch import nn
from torch.nn import init
from torch import functional as F

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

class LinearRegression(nn.Module, MyModelMixin):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=True):
        super(LinearRegression, self).__init__()
        torch.manual_seed(111)
        self.linear = Linear(input_dim, output_dim, bias)

    def forward(self, X, **kwargs):
        return self.linear(X)

    def get_weights(self):
        return self.linear.get_weights()

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

class SparseFeatureLinearRegression(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=False):
        super(SparseFeatureLinearRegression, self).__init__()
        torch.manual_seed(111)
        self.input_mask = nn.Parameter(torch.zeros([1, input_dim]).normal_(0, 1))
        self.output = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, **kwargs):
        X = (X * self.input_mask)
        X = self.output(X)
        return X, X

    def get_weights(self):
        return {
            "weights": self.output.weight * self.input_mask
        }


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=4096,
            bias=False
    ):
        super(MLP, self).__init__()
        torch.manual_seed(111)
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.output = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, X, **kwargs):
        X = torch.relu(self.hidden(X))
        #X = self.dropout(X)
        X = self.output(X)
        return X

    def get_weights(self):
        return {
            "output_weights": self.output.weight,
            "hidden_weights": self.hidden.weight,
        }



class SparseWeightNet(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
    ):
        super(SparseWeightNet, self).__init__()
        torch.manual_seed(111)

        self.input_layer = SpaRedLinear(input_dim, hidden_dim)
        # self.hidden_layer = SpaRedLinear(hidden_dim, hidden_dim)
        self.output_layer = SpaRedLinear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = torch.relu(self.input_layer(X))
        #X = torch.relu(self.hidden_layer(X))
        X = self.output_layer(X)
        return X

    def get_weights(self):
        return {'input_weights': self.input_layer.get_weights()['weight'],
                #'hidden_weights': self.hidden_layer.get_weights()['weight'],
                'output_weights': self.output_layer.get_weights()['weight']}


class SparseFeatureNet(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=200,
    ):
        super(SparseFeatureNet, self).__init__()
        torch.manual_seed(111)
        self.input = SparseFeatureLinearRegression(input_dim, hidden_dim)
        self.output = SpaRedLinear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = torch.relu(self.input(X))
        X = self.output(X)
        return X, X

    def get_weights(self):
        return {'input_weights': self.input.get_weights()['weights'],
                'output_weights': self.output.weight}


class SparseFeatureNetv2(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=1024,
    ):
        super(SparseFeatureNetv2, self).__init__()
        torch.manual_seed(111)
        self.input_mask = nn.Parameter(torch.zeros([1, input_dim]).normal_(0, 1))
        self.linear_output = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_feature = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mlp_output = nn.Sequential(
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X, **kwargs):
        sparse_feature = self.input_mask * X
        X1 = self.linear_output(sparse_feature)
        X2 = self.linear_feature(sparse_feature)
        X2 = self.mlp_output(X2)
        return X1, X2

    def get_weights(self):
        return {'input_mask': self.input_mask}
