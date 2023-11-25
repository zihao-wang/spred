# from args import args
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        # if first_layer:
        #     print(f"==> Building first layer with {args.first_layer_type}")

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        self._init_conv(conv)

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)

    def activation(self):
        # if args.nonlinearity == "relu":
        #     return (lambda: nn.ReLU(inplace=True))()
        # else:
        #     raise ValueError(f"{args.nonlinearity} is not an initialization option!")
        return (lambda: nn.ReLU(inplace=True))()

    def _init_conv(self, conv):
        # if args.init == "signed_constant":

        #     fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
        #     if args.scale_fan:
        #         fan = fan * (1 - args.prune_rate)
        #     gain = nn.init.calculate_gain(args.nonlinearity)
        #     std = gain / math.sqrt(fan)
        #     conv.weight.data = conv.weight.data.sign() * std

        # elif args.init == "unsigned_constant":

        #     fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
        #     if args.scale_fan:
        #         fan = fan * (1 - args.prune_rate)

        #     gain = nn.init.calculate_gain(args.nonlinearity)
        #     std = gain / math.sqrt(fan)
        #     conv.weight.data = torch.ones_like(conv.weight.data) * std

        # elif args.init == "kaiming_normal":

        #     if args.scale_fan:
        #         fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
        #         fan = fan * (1 - args.prune_rate)
        #         gain = nn.init.calculate_gain(args.nonlinearity)
        #         std = gain / math.sqrt(fan)
        #         with torch.no_grad():
        #             conv.weight.data.normal_(0, std)
        #     else:
        #         nn.init.kaiming_normal_(
        #             conv.weight, mode=args.mode, nonlinearity=args.nonlinearity
        #         )

        # elif args.init == "standard":
        #     nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        # else:
        #     raise ValueError(f"{args.init} is not an initialization option!")

        # if args.scale_fan:
        #     fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
        #     fan = fan * (1 - args.prune_rate)
        #     gain = nn.init.calculate_gain(args.nonlinearity)
        #     std = gain / math.sqrt(fan)
        #     with torch.no_grad():
        #         conv.weight.data.normal_(0, std)
        # else:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_in", nonlinearity="relu"
        )

class VanillaConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thr = -1

    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        if self.thr > 0:
            sparse_weight = self.weight.clone()
            mask = sparse_weight.abs().lt(self.thr)
            sparse_weight[mask] = 0
        x = F.conv2d(
            x, sparse_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, t=1e-3):
        nonzero = self.weight.abs().gt(t).sum()
        total = self.weight.numel()
        return nonzero, total, t

class SpredConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.red_weight = nn.Parameter(self.weight.clone())
        self.thr = -1

    def forward(self, x):
        if self.thr > 0:
            sparse_weight = self.red_weight * self.weight
            mask = sparse_weight.abs().lt(self.thr)
            sparse_weight[mask] = 0
        x = F.conv2d(
            x, sparse_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, t=1e-3):
        sparse_weight = self.red_weight * self.weight
        nonzero = sparse_weight.abs().gt(t).sum()
        total = sparse_weight.numel()
        return nonzero, total, t


def get_builder(conv_type="VanillaConv"):
    if conv_type == "VanillaConv":
        conv_layer = VanillaConv
    else:
        conv_layer = SpredConv

    first_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=nn.BatchNorm2d, first_layer=first_layer)

    return builder
