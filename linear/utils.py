import json
import numpy as np
import os
import time

import torch

from models import MyModelMixin

def get_model_device(model):
    # Check the device of the first parameter (usually the first layer's parameter)
    return next(model.parameters()).device

def eval_over_linear_regression_datasets(x, y, trans, alpha):
    _, predictor_dim = x.shape

    _, respond_dim = y.shape
    # trans = trans.reshape((predictor_dim, respond_dim))
    if isinstance(trans, torch.nn.Module):
        trans.eval()
        device = get_model_device(trans)
        _x = torch.tensor(x, device=device, dtype=torch.float32)
        _y = torch.tensor(y, device=device, dtype=torch.float32)
        _diff = _y - trans(_x)
        diff = _diff.detach().cpu().numpy()
    else:
        diff = y - x.dot(trans)

    mse = np.linalg.norm(diff ** 2, axis=-1, ord=2).mean() / 2
    if isinstance(trans, torch.nn.Module):
        l1 = 0
        for k, w in trans.get_weights().items():
            l1 += torch.norm(w, p=1).item()
        w = w.detach().cpu().numpy()
    else:
        l1 = np.linalg.norm(trans, ord=1)
        w = trans

    zero_rate3 = (np.abs(w) < 1e-3).mean()
    zero_rate6 = (np.abs(w) < 1e-6).mean()
    zero_rate9 = (np.abs(w) < 1e-9).mean()
    zero_rate12 = (np.abs(w) < 1e-12).mean()
    ret = {'mse': mse, 'l1': l1, 'total': mse + l1 * alpha,
           'zero_rate3': zero_rate3,
           'zero_rate6': zero_rate6,
           'zero_rate9': zero_rate9,
           'zero_rate12': zero_rate12}
    return {k: float(v) for k, v in ret.items()}
