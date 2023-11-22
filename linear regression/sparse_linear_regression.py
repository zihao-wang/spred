import argparse
import os
from collections import defaultdict
import pickle

import numpy as np

np.set_printoptions(precision=8, suppress=True)
import pandas as pd

from utils import eval_over_linear_regression_datasets
from routines import run_l1_regression, run_lasso, run_rs_regression, run_rs_regression_v2
from data import isotropic_predictor_data

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--predictor_dim', type=int, default=100)
parser.add_argument('--respond_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--noisy_variance', type=float, default=0.1)
parser.add_argument('--optname', type=str, default='SGD')
parser.add_argument('--regularization', type=str, default='rs', choices=['lasso', 'rs', 'l1'])
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_alpha', type=int, default=20)
parser.add_argument('--alpha', type=float, default=-1)
parser.add_argument('--output_folder', type=str, default='output')


def run_alpha(alpha, args, metric_output={}):
    if args.regularization == 'lasso':
        fetch = run_lasso(alpha, x, y)
    elif args.regularization == 'l1':
        fetch = run_l1_regression(alpha, x, y,
                                  args.optname,
                                  args.batch_size,
                                  args.lr,
                                  args.epoch,
                                  args.device)
    elif args.regularization == 'rs':
        fetch = run_rs_regression_v2(alpha, x, y,
                                     optname=args.optname,
                                     batch_size=args.batch_size,
                                     lr=args.lr,
                                     epochs=args.epoch,
                                     device=args.device)


    eval_fetch = eval_over_linear_regression_datasets(x, y, fetch['weights'], alpha)
    for k in eval_fetch:
        metrics_output[k].append(eval_fetch[k])
    metrics_output['time'].append(fetch['time'])
    metrics_output['method'].append(args.regularization)
    metrics_output['train_fetch'].append(fetch)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_folder, exist_ok=True)

    (x, y), trans = isotropic_predictor_data(args.num_samples,
                                             args.predictor_dim,
                                             args.respond_dim,
                                             args.noisy_variance)

    # lasso_weights_sorted = defaultdict(list)
    # l1_weights_sorted = defaultdict(list)
    # rs_weights_sorted = defaultdict(list)
    metrics_output = defaultdict(list)

    if args.alpha > 0:
        alpha = args.alpha
        run_alpha(alpha, args, metrics_output)
    else:
        # manually set the alpha thresholding
        # adding a small value to ensure the soft threshold
        non_zero_coefficient = trans[np.nonzero(trans)] + 1e-3
        alpha_range = np.abs(non_zero_coefficient).ravel().tolist()
        alpha_range.sort()

        alpha_range = np.arange(args.num_alpha) / (args.num_alpha - 1) * \
                    np.max(np.abs(non_zero_coefficient)) * 1.1


        for alpha in alpha_range.tolist():
            print('alpha = ', alpha)
            metrics_output['alpha'].append(alpha)
            run_alpha(alpha, args, metrics_output)

    with open(os.path.join(args.output_folder, 'results.pickle'), 'wb') as f:
        pickle.dump(metrics_output, f)

    for k in metrics_output:
        if 'train' in k:
            continue
        print(k, metrics_output[k])