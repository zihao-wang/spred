import argparse
import json
import logging
import os

import torch
from torch import nn

import data
from torch_saga import SAGA
from models import SparseFeatureLinearRegression
from utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=10000)
parser.add_argument('--num_steps', type=int, default=1000000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--alpha', type=float, default=1e-5)
parser.add_argument('--logging_path', type=str, default="output/SparseLogisticRegression/test.log")



def neural_train_and_compress_ratio(dataset_callback,
                                    num_steps=1000000,
                                    lr=0.004,
                                    alpha=1e-3,
                                    thr=1e-10,
                                    device="cuda:0"):
    arrlist, (input_dim, output_dim) = dataset_callback()
    X_train, X_test, y_train, y_test = [
        torch.from_numpy(arr).to(device) for arr in arrlist]
    net = SparseFeatureLinearRegression(input_dim, output_dim, False).to(device)
    # net = LogisticRegression(input_dim, output_dim, bias=False).to(device)
    loss_func = nn.CrossEntropyLoss()

    trainer = SAGA(net, loss_func, reg_p=2, alpha=alpha)
    trainer.run(
        X_train, y_train, lr, num_steps,
        log_callback=Logger(
            log_file_path=args.logging_path,
            X_test=X_test,
            y_test=y_test,
            thr=thr)
    )


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

    fetch = neural_train_and_compress_ratio(
        dataset_cbk,
        num_steps=args.num_steps,
        lr=args.lr,
        alpha=args.alpha,
        device=args.device
    )