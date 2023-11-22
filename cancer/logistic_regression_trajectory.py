import argparse
from collections import defaultdict
import time
import json
import logging
import os
from typing import Callable

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from skorch import NeuralNetClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data
from models import SpaRedLinear

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--alpha', type=float, default=1e-5)
parser.add_argument('--logging_path', type=str, default="output/SparseLogisticRegression")


class SparseLinear(nn.Module):
    def __init__(
            self,
            input_dim=130107,
            output_dim=20):
        super(SparseLinear, self).__init__()
        self.output = SpaRedLinear(input_dim, output_dim)

    def forward(self, X, *args, **kwargs):
        #X = self.dropout(X)
        X = torch.softmax(self.output(X), dim=-1)
        return X

    def get_weights(self):
        return self.output.get_weights()


def run_time_accuracy_trajectory_nn(dataset_callback: Callable,
                                    max_epoch=10,
                                    lr=0.004,
                                    alpha=1e-5,
                                    device='cuda:0'
                                    ):
    X_train, X_test, y_train, y_test = dataset_callback()

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    input_dim = X_train.size(1)
    if input_dim < 1000:
        output_dim = 10
    else:
        output_dim = 20

    model = SparseLinear(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    loss_func = nn.CrossEntropyLoss()

    epoch_metrics = defaultdict(list)
    t0 = time.time()
    for e in range(max_epoch):
        for X, y in dataloader:

            y_pred = model(X)
            loss = loss_func(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_pred = model(X_test)
        accuracy = (y_pred.argmax(-1) == y_test).float().mean().item()
        epoch_metrics["test_accuracy"].append(accuracy)
        epoch_metrics["epoch"].append(e+1)
        epoch_metrics["time"].append(time.time() - t0)

    return epoch_metrics

def run_time_accuracy_trajectory_sklearn(dataset_callback: Callable,
                                    max_epoch=10,
                                    alpha=1e-5):

    X_train, X_test, y_train, y_test = dataset_callback()

    epoch_metrics = defaultdict(list)
    for e in range(max_epoch):
        t0 = time.time()
        model = LogisticRegression(penalty='l1',
                                   C=1/alpha,
                                   max_iter=e+1,
                                   multi_class='multinomial',
                                   solver='saga')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        epoch_metrics["test_accuracy"].append(accuracy)
        epoch_metrics["epoch"].append(e+1)
        epoch_metrics["time"].append(time.time() - t0)

    return epoch_metrics


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.logging_path, exist_ok=True)
    filename = "dataset={}_alpha={}.csv".format(args.dataset, args.alpha)

    dataset_callback = getattr(data, 'get_'+args.dataset)
    data = {}
    epoch_metrics = run_time_accuracy_trajectory_nn(
        dataset_callback,
        max_epoch=args.epoch,
        lr=args.lr,
        alpha=args.alpha,
        device=args.device)
    for k in epoch_metrics:
        data["nn:" + k] = epoch_metrics[k]
    epoch_metrics = run_time_accuracy_trajectory_sklearn(
        dataset_callback,
        max_epoch=args.epoch,
        alpha=args.alpha
    )
    for k in epoch_metrics:
        data["lasso:" + k] = epoch_metrics[k]

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.logging_path, filename), index=False)
