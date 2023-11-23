import argparse
import os
from collections import defaultdict
import json

import numpy as np
import pandas as pd

from utils import eval_over_linear_regression_datasets
from routines import run_lasso, run_rs_regression
from data import isotropic_predictor_data

np.set_printoptions(precision=8, suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--predictor_dim', type=int, default=100)
parser.add_argument('--respond_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--noisy_variance', type=float, default=0.1)
parser.add_argument('--optname', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_alpha', type=int, default=10)

def get_closest_alpha(alphas, a):
    min_idx = np.abs(a - np.asarray(alphas)).argmin()
    return alphas[min_idx]

if __name__ == "__main__":
    args = parser.parse_args()
    output_folder = os.path.join("output/HighDimLinearRegression-v2",
                                 f"{args.predictor_dim}_{args.respond_dim}")
    os.makedirs(output_folder, exist_ok=True)

    (x, y), trans = isotropic_predictor_data(args.num_samples,
                                             args.predictor_dim,
                                             args.respond_dim,
                                             args.noisy_variance,
                                             seed=666)

    # manually set the alpha thresholding
    alpha_range = np.logspace(-9, 1, args.num_alpha, base=3)
    alpha_range = np.floor(alpha_range * 1e6 ) * 1e-6
    print(alpha_range)
    # exit()

    # run lars if there is no lasso record
    lars_file = os.path.join(output_folder, 'lars_metrics.csv')
    # re run the lasso
    if os.path.exists(lars_file):
        print(f"lasso file {lars_file} already found")
        lars_df = pd.read_csv(lars_file)
    else:
        print(f"lasso file {lars_file} not found")
        data = defaultdict(list)
        for alpha in alpha_range.tolist():
            data['alpha'].append(alpha)

            lars_fetch = run_lasso(alpha, x, y, method='LARS')
            data['time'].append(lars_fetch['time'])
            lars_eval_fetch = eval_over_linear_regression_datasets(x, y, lars_fetch['weights'], alpha)
            for k in lars_eval_fetch:
                data[k].append(lars_eval_fetch[k])
        lars_df = pd.DataFrame(data)
        lars_df.to_csv(lars_file, index=False)
        print(lars_df.to_string())

    # run lasso if there is no lasso record
    lasso_file = os.path.join(output_folder, 'lasso_metrics.csv')
    # re run the lasso
    if os.path.exists(lasso_file):
        print(f"lasso file {lasso_file} already found")
        lasso_df = pd.read_csv(lasso_file)
    else:
        print(f"lasso file {lasso_file} not found")
        data = defaultdict(list)
        for alpha in alpha_range.tolist():
            data['alpha'].append(alpha)

            lasso_fetch = run_lasso(alpha, x, y, method='lasso')
            data['time'].append(lasso_fetch['time'])
            lasso_eval_fetch = eval_over_linear_regression_datasets(x, y, lasso_fetch['weights'], alpha)
            for k in lasso_eval_fetch:
                data[k].append(lasso_eval_fetch[k])
        lasso_df = pd.DataFrame(data)
        lasso_df.to_csv(lasso_file, index=False)
        print(lasso_df.to_string())

        lasso_file = os.path.join(output_folder, 'lasso_metrics.csv')
        lasso_df = pd.read_csv(lasso_file)


    # run rs
    data = defaultdict(list)
    for alpha in alpha_range.tolist():
        a = get_closest_alpha(lasso_df.alpha, alpha)
        lasso_record = lasso_df[lasso_df.alpha == a].to_dict('list')
        print(lasso_record)
        target_loss = lasso_record['total'][0]
        target_zero_rate = lasso_record['zero_rate12'][0]
        rs_fetch = run_rs_regression(a, x, y,
                                     optname=args.optname,
                                     epochs=args.epoch,
                                     batch_size=args.batch_size,
                                     lr=args.lr,
                                     device=args.device,
                                     loss_func='mse',
                                     loss_less_than=target_loss,
                                     zero_rate_greater_than=target_zero_rate,
                                     zero_rate_ratios=[0.5, 0.75, 0.9, 0.99, 0.999, 1],
                                     eval_every_epoch=100)

        filename = os.path.join(
            output_folder,
            f'{alpha}-rs_metrics_optname_{args.optname}_lr_{args.lr}')

        with open(filename, 'wt') as f:
            for metric in rs_fetch['metric_list']:
                f.write(json.dumps(metric) + '\n')
