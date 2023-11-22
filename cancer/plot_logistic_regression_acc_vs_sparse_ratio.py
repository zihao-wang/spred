from collections import defaultdict
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt

def read_log(dataset, method):
    logname = os.path.join("output", "SparseLogisticRegression", f"{dataset}_{method}.log")
    data = defaultdict(list)
    with open(logname, 'rt') as f:
        for line in f.readlines():
            if "root" not in line:
                continue
            info = line[10:]
            try:
                d = eval(info)
            except:
                continue
            for k in d:
                data[k].append(d[k])
    df = pd.DataFrame(data)
    df = df.sort_values(by="compress_ratio")
    return df

def plot_dataset(dataset):
    plt.figure(figsize=(3, 2.5))
    for method in method_list:
        df = read_log(dataset, method)
        plt.plot(df.compress_ratio, df.accuracy, label=method)
    plt.legend()
    plt.xlabel("compress ratio")
    plt.ylabel("test accuracy")
    plt.xscale('log')
    plt.title(dataset)
    plt.tight_layout()
    plt.savefig(f"output/plots/logistic_regression_acc_vs_sparse_ratio/{dataset}.png")
    plt.savefig(f"output/plots/logistic_regression_acc_vs_sparse_ratio/{dataset}.pdf")


if __name__ == "__main__":
    dataset_list = ["20news", "mnist"]
    method_list = ["lr", "sparse_weight"]

    for dataset in dataset_list:
        plot_dataset(dataset)