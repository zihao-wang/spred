import argparse
import json
import logging
import os

import torch
from sklearn.linear_model import LogisticRegression
from skorch import NeuralNetClassifier
from torch import nn

import data
from models import SpaRedLinear

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--alpha', type=float, default=1e-5)
parser.add_argument('--logging_path', type=str, default="output/SparseLogisticRegression/log")


def classic_train_and_compress_ratio(dataset_callback,
                                     solver,
                                     multi_class,
                                     alpha,
                                     thr=1e-10):
    X_train, X_test, y_train, y_test = dataset_callback()

    clf = LogisticRegression(
        penalty='l1',
        C=1/alpha,
        solver=solver,
        multi_class=multi_class,
        max_iter=10
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    weights = clf.coef_
    total_num = weights.shape[0] * weights.shape[1]
    non_zero_num = (weights > thr).sum()
    compress_ratio = total_num / non_zero_num
    print(accuracy, compress_ratio)
    return {'accuracy': accuracy,
            'compress_ratio': compress_ratio}


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.logging_path),
        exist_ok=True
    )
    logging.basicConfig(
        filename=args.logging_path,
        level=logging.INFO,
        filemode='at'
    )

    dataset_cbk = getattr(data, 'get_' + args.dataset)

    fetch = classic_train_and_compress_ratio(
        dataset_cbk,
        solver='saga',
        multi_class='multinomial',
        alpha=args.alpha
    )

    fetch['alpha'] = args.alpha
    logging.info(fetch)
