import argparse
import os
import logging
from secrets import choice
import time

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from pyHSICLasso import HSICLasso

from data import get_cancer_GDS
from models import MLP, SparseFeatureLinearRegression, SparseFeatureNet, SparseFeatureNetv2, SparseWeightNet
from routines import run_classification, run_sparse_feature_classification


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default='hsiclasso')
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--log_path", type=str, default='log/default.log')
# hsic lasso arguments
parser.add_argument("--num_feat", type=int, default=50)
parser.add_argument("--B", type=int, default=20)
# spared arguments
parser.add_argument("--alpha", type=float, default=1e-3)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--optname", type=str, default="Adam")


def _logistic_regression(X_train, y_train, X_test, y_test, **kwargs):
    clf = LogisticRegression(
        penalty='l1', solver='saga', C=100, max_iter=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    return {'acc': acc}

def _hsic_lasso(X_train, y_train, X_test, y_test, num_feat, B, **kwargs):
    hsic_lasso = HSICLasso()
    hsic_lasso.input(X_train, y_train)
    hsic_lasso.classification(num_feat=num_feat, B=B)
    feat_index = hsic_lasso.get_index()
    X_train_selected = X_train[:, feat_index]
    X_test_selected = X_test[:, feat_index]
    # first pick the variables
    clf = SVC()
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_pred, y_test)
    return {'acc': acc}

def _some_net(NetClass, X_train, y_train, X_test, y_test, alpha, device, **kwargs):
    input_dim = X_train.shape[1]
    output_dim = max(np.max(y_train), np.max(y_test)) + 1
    net = NetClass(input_dim, output_dim).to(device)
    acc, metric_list = run_classification(
        alpha, X_train, y_train, X_test, y_test, net,
        device=device, **kwargs)
    return {'acc': acc}

def _sparse_feature_svm(NetClass, X_train, y_train, X_test, y_test, alpha, device, **kwargs):
    input_dim = X_train.shape[1]
    output_dim = max(np.max(y_train), np.max(y_test)) + 1
    net = NetClass(input_dim, output_dim).to(device)
    acc, final_features, metric_list = run_sparse_feature_classification(
        alpha, X_train, y_train, X_test, y_test, net,
        device=device, **kwargs)
    mask = (net.input_mask > 1e-10).cpu().detach().numpy().reshape(-1)
    X_train_selected = X_train[:, mask]
    X_test_selected = X_test[:, mask]
    clf = SVC()
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_pred, y_test)
    return {'acc': acc, 'final_features': mask.sum().item()}

def _sparse_feature_net(NetClass, X_train, y_train, X_test, y_test, alpha, device, **kwargs):
    input_dim = X_train.shape[1]
    output_dim = max(np.max(y_train), np.max(y_test)) + 1
    net = NetClass(input_dim, output_dim).to(device)
    mask = (net.input_mask > 1e-10).cpu().detach().numpy().reshape(-1)
    acc, final_features, metric_list = run_sparse_feature_classification(
        alpha, X_train, y_train, X_test, y_test, net,
        device=device, **kwargs)
    return {'acc': acc, 'final_features': mask.sum().item()}

def evaluate_model(X_train, y_train, X_test, y_test, model_name, **kwargs):
    if model_name.lower() == 'logistic_regression':
        """logistic regression with L1 penalty"""
        metric = _logistic_regression(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'hsiclasso':
        """first run hsic lasso then use the rbf svm classifier"""
        metric = _hsic_lasso(X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'sparse_feature_linear':
        """run the sparse feature logistic regression"""
        metric = _sparse_feature_net(SparseFeatureLinearRegression, X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'sparse_feature_linear_svm':
        """first run the sparse feature logistic regression and then pick the feature for svc"""
        metric = _sparse_feature_svm(SparseFeatureLinearRegression, X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'mlp':
        """run a mlp"""
        metric = _some_net(MLP, X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'sparse_feature_net':
        """run a joint trained feature selection and end to end network"""
        metric = _sparse_feature_net(SparseFeatureNet, X_train, y_train, X_test, y_test, **kwargs)
    elif model_name.lower() == 'sparse_feature_net_v2':
        """run a joint trained feature selection and end to end network"""
        metric = _sparse_feature_net(SparseFeatureNetv2, X_train, y_train, X_test, y_test, **kwargs)
    else:
        raise NotImplementedError
    return metric

def multi_trials_model(X, y, model_name, num_trials=100, **kwargs):
    acc_list = []
    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)
        t0 = time.time()
        metric = evaluate_model(X_train, y_train, X_test,
                             y_test, model_name, **kwargs)
        dt = time.time() - t0
        metric['dt'] = dt
        logging.info(f"random trail {i+1} of {num_trials}:{metric}")
        acc_list.append(metric['acc'])
    return {
        'model_name': model_name,
        'mean_acc': np.mean(acc_list),
        'var_acc': np.std(acc_list),
    }


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logging.basicConfig(filename=args.log_path,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='wt',
                        level=logging.INFO)

    logging.info(args)
    np.random.seed(111)
    torch.manual_seed(111)

    for f in os.listdir('data'):
        if f.endswith('soft.gz'):
            filepath = os.path.join("data", f)
            X, y = get_cancer_GDS(filepath)
            perm = np.random.permutation(X.shape[0])
            X = X[perm, :]
            y = y[perm]
            metric = multi_trials_model(X, y, **vars(args))
            logging.info("---- Begin of Report ----")
            logging.info(args)
            logging.info(f)
            metric['dataset'] = f
            metric.update(vars(args))
            logging.info(f"metric:{metric}")
            logging.info("---- End of Report ----")
