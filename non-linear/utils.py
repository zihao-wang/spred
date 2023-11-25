import json
import numpy as np
import os
import time

from models import MyModelMixin


def eval_over_linear_regression_datasets(x, y, trans, alpha):
    _, predictor_dim = x.shape

    _, respond_dim = y.shape
    # trans = trans.reshape((predictor_dim, respond_dim))
    diff = y - x.dot(trans)

    mse = np.linalg.norm(diff ** 2, axis=-1, ord=2).mean() / 2
    l1 = np.linalg.norm(trans, ord=1)

    zero_rate3 = (np.abs(trans) < 1e-3).mean()
    zero_rate6 = (np.abs(trans) < 1e-6).mean()
    zero_rate9 = (np.abs(trans) < 1e-9).mean()
    zero_rate12 = (np.abs(trans) < 1e-12).mean()
    ret = {'mse': mse, 'l1': l1, 'total': mse + l1 * alpha,
           'zero_rate3': zero_rate3,
           'zero_rate6': zero_rate6,
           'zero_rate9': zero_rate9,
           'zero_rate12': zero_rate12}
    return {k: float(v) for k, v in ret.items()}


class Logger:
    def __init__(self, log_file_path, X_test, y_test, thr=1e-10):
        self.log_file_path = log_file_path
        os.remove(self.log_file_path)
        self.X_test = X_test
        self.y_test = y_test
        self.thr = thr

    def write(self, info):
        with open(self.log_file_path, 'at') as f:
            f.write(info + '\n')

    def __call__(self, net: MyModelMixin):
        t = time.time()

        # test accuracy
        logits = net(self.X_test)
        y_pred = logits.argmax(-1)
        acc = (y_pred == self.y_test).float().mean().item()

        weight_dict = net.get_weights()
        total_num_w = 0
        num_zero_w = 0
        for _, ten in weight_dict.items():
            total_num_w += ten.numel()
            num_zero_w += (ten < self.thr).float().mean().item()
        comp_ratio = total_num_w / (total_num_w - num_zero_w)

        log = {
            "time": t,
            "acc": acc,
            "compression_ratio": comp_ratio,
        }

        self.write(json.dumps(log))
