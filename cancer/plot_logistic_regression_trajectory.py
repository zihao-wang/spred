from collections import defaultdict
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt

def plot_trajectory_df(filename):
    print(filename)

    df = pd.read_csv(os.path.join(datadir, filename))
    plt.figure(figsize=(3, 2.5))
    plt.plot(df['nn:time'], df['nn:test_accuracy'], label='sparse weight')
    plt.plot(df['lasso:time'], df['lasso:test_accuracy'], label='saga solver')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("test accuracy")
    plt.savefig(os.path.join(plotdir, filename.replace(".csv", ".png")))
    plt.savefig(os.path.join(plotdir, filename.replace(".csv", ".pdf")))

if __name__ == "__main__":
    plotdir = "output/plots/logistic_regression_trajectory"
    os.makedirs(plotdir, exist_ok=True)
    datadir = "output/SparseLogisticRegression"
    for file in os.listdir(datadir):
        if file.endswith('.csv'):
            plot_trajectory_df(file)