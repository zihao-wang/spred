import os
import pickle
import os.path as osp
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

methods = ['lasso', 'rs']
input_dims = [100, 1000, 10000, 100000]
output_dims = [1, 100]
alphas = [1, 2, 3, 4, 5]
zr_ratios = [0.75, 0.9, 0.99]
zr_markers = ['<', '^', '>', 's', 'o']


def get_rs_min_time(input_dim, output_dim, alpha, criteria):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_time = None
    for fn in os.listdir(folder):
        if 'rs' in fn and not fn.endswith('.csv') and fn.startswith(f'{alpha}-'):
            print(fn)
            fp = osp.join(folder, fn)
            t = None
            with open(fp, 'rt') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    if record[criteria]:
                        t = record['time']
                        break

    if min_time:
        min_time = min(min_time, t)
    else:
        min_time = t
    return min_time


def get_rs_min_loss(input_dim, output_dim, alpha, criteria):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_time = None
    for fn in os.listdir(folder):
        if 'rs' in fn and not fn.endswith('.csv') and fn.startswith(f'{alpha}-'):
            print(fn)
            fp = osp.join(folder, fn)
            t = None
            with open(fp, 'rt') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    if record[criteria]:
                        t = record['total']
                        break

    if min_time:
        min_time = min(min_time, t)
    else:
        min_time = t
    return min_time


def get_lasso_time(input_dim, output_dim, method, alpha):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_time = None
    t = None
    for fn in os.listdir(folder):
        if method.lower() in fn:
            fp = osp.join(folder, fn)
            df = pd.read_csv(fp)
            record = df[df.alpha == alpha].to_dict('list')
            t = record['time'][0]
    if min_time:
        min_time = min(min_time, t)
    else:
        min_time = t
    return min_time


def get_lasso_loss(input_dim, output_dim, method, alpha):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_loss = None
    t = None
    for fn in os.listdir(folder):
        if method.lower() in fn:
            fp = osp.join(folder, fn)
            df = pd.read_csv(fp)
            record = df[df.alpha == alpha].to_dict('list')
            t = record['total'][0]
    if min_loss:
        min_loss = min(min_loss, t)
    else:
        min_loss = t
    return min_loss


def plot_time_vs_inputdim():
    for a in alphas:
        for outdim in output_dims:
            plt.figure(figsize=(5, 2.5))
            y = [get_lasso_time(indim, outdim, 'lasso', a) for indim in input_dims]
            plt.plot(input_dims, y, "r-", label='Lasso')

            y = [get_lasso_time(indim, outdim, 'lars', a) for indim in input_dims]
            plt.plot(input_dims, y, "y-", label='LARS')

            for _c, _m in zip(zr_ratios, zr_markers):
                cri = f'zero_rate_greater_than_threshold:{_c}'
                y = [get_rs_min_time(indim, outdim, a, cri) for indim in input_dims]
                if y[0] is None or y[1] is None:
                    continue
                plt.plot(input_dims, y, "b-", marker=_m, alpha=0.5, label=f'SR {_c}')

            plt.xlabel('Input Dimension')
            plt.ylabel('Time')
            plt.xscale('log')
            plt.yscale('log')
            # plt.title(rf'$\alpha={a / 5} \alpha_m$, Output dimension={outdim}')
            plt.legend(bbox_to_anchor=(1.1, 1))
            plt.tight_layout()
            plt.savefig(f"output/plots/regression-general/alpha={a}_output_dim={outdim}.pdf")
            plt.savefig(f"output/plots/regression-general/alpha={a}_output_dim={outdim}.png")

def plot_time_vs_alpha():
    x = np.asarray(alphas)
    x = x / np.max(alphas)
    for outdim in output_dims:
        for indim in input_dims:
            plt.figure(figsize=(5, 2.5))
            y = [get_lasso_time(indim, outdim, 'lasso', a) for a in alphas]
            plt.plot(x, y, "r-", label='Lasso')

            y = [get_lasso_time(indim, outdim, 'lars', a) for a in alphas]
            plt.plot(x, y, "y-", label='LARS')

            for _c, _m in zip(zr_ratios, zr_markers):
                cri = f'zero_rate_greater_than_threshold:{_c}'
                y = [get_rs_min_time(indim, outdim, a, cri) for a in alphas]
                if y[0] is None or y[1] is None:
                    continue
                plt.plot(x, y, "b-", marker=_m, alpha=0.5, label=f'SR {_c}')

            plt.xlabel(r'$\alpha/\alpha_m$')
            plt.ylabel('Time')
            plt.yscale('log')
            # plt.title(rf'Input dimension={indim}, Output dimension={outdim}')
            plt.legend(bbox_to_anchor=(1.1, 1))
            plt.tight_layout()
            plt.savefig(f"output/plots/regression-general/indim={indim}_output_dim={outdim}.pdf")
            plt.savefig(f"output/plots/regression-general/indim={indim}_output_dim={outdim}.png")


if __name__ == '__main__':

    os.makedirs('output/plots/regression-general', exist_ok=True)
    plot_time_vs_alpha()
    plot_time_vs_inputdim()
