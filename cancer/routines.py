import logging
from collections import Counter
import time

import torch
import numpy as np
from tqdm import trange
from sklearn.linear_model import Lasso, LassoLars
from sklearn.model_selection import train_test_split

from models import LinearRegression, SparedLinearRegression, SparseFeatureNet, SparseFeatureNetv2
from utils import eval_over_linear_regression_datasets


class EarlyEscapeZeroRate:
    def __init__(self, patience, beta=0.99) -> None:
        self.patience = patience
        self.beta = beta
        self.past_zero_rate = -1

    def check_escape(self, zero_rate):
        self.past_zero_rate = self.beta * \
                              self.past_zero_rate + (1 - self.beta) * zero_rate

        if self.past_zero_rate >= zero_rate:
            return True
        else:
            return False


def run_lasso(alpha, x, y, method='default', **kwargs):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape

    if method == 'LARS':
        lasso_regressor = LassoLars(alpha=alpha, normalize=False, max_iter=100000)
    else:
        lasso_regressor = Lasso(alpha=alpha, max_iter=100000)

    t = time.time()
    lasso_regressor.fit(x, y)
    t = time.time() - t

    _coef = lasso_regressor.coef_.reshape(respond_dim, predictor_dim)
    weights = _coef.T

    return {'time': t,
            'weights': weights}

def run_rs_regression_v2(alpha, x, y,
                      optname='SGD',
                      epochs=200,
                      batch_size=512,
                      lr=1e-4,
                      device='cuda:0',
                      tol=1e-6,
                      eval_every_epoch=100):

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    _, predictor_dim = x.shape
    _, respond_dim = y.shape
    model = SparedLinearRegression(input_dim=predictor_dim,
                                   output_dim=respond_dim,
                                   bias=True)
    # using weight decay for L2 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr, weight_decay=alpha)

    metric_list = []
    last_loss = -1

    t = time.time()
    with trange(epochs) as titer:
        for e in titer:
            metric = {}
            total_loss = 0
            for x_batch, y_batch in dataloader:
                y_pred = model(x_batch)
                weight_dict = model.get_weights()
                l1_reg = 0
                loss = 0
                for k, w in weight_dict.items():
                    l1_reg += torch.norm(w, p=1)
                _func = torch.nn.MSELoss()
                loss += _func(y_pred, y_batch) / 2
                assert not torch.isnan(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() + alpha * l1_reg.item()

            epoch_loss = total_loss / len(dataloader)
            metric['epoch_loss'] = epoch_loss
            metric['epoch'] = e + 1
            metric['time'] = time.time() - t


            if e % eval_every_epoch == 0:
                zero6 = 0
                zero12 = 0
                total = 0
                this_loss = loss.item()
                weight_dict = model.get_weights()
                for k, w in weight_dict.items():
                    total += w.numel()
                    zero6 = torch.sum(torch.abs(w) < 1e-6)
                    zero12 = torch.sum(torch.abs(w) < 1e-12)

                sparse6 = zero6 / total
                sparse12 = zero12 / total

                metric['sparse6'] = sparse6
                metric['sparse12'] = sparse12
                err1 = torch.abs(sparse6 - sparse12)
                err2 = np.abs(this_loss - last_loss)
                if (err1 < tol) and (err2 < tol):
                    print(err1, err2)
                    break

                last_loss = this_loss

    return {'time': time.time() - t,
            'weights': model.get_weights()['weight'].cpu().detach().numpy().T,
            'metric_list': metric_list}


def run_rs_regression(alpha, x, y,
                      net=None,
                      optname='SGD',
                      epochs=200,
                      batch_size=512,
                      lr=1e-4,
                      loss_func='ce',
                      device='cuda:0',
                      loss_less_than=0,
                      zero_rate_greater_than=1,
                      zero_rate_ratios=[0.5, 0.75, 0.9, 0.99, 0.999, 1, 1.01],
                      eval_every_epoch=100, **kwargs):

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    if loss_func == 'ce':
        y_tensor = torch.tensor(y, dtype=torch.int64, device=device)
    elif loss_func == 'mse':
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    if net is None:
        _, predictor_dim = x.shape
        _, respond_dim = y.shape
        model = SparedLinearRegression(input_dim=predictor_dim,
                                       output_dim=respond_dim)
    else:
        model = net

    # using weight decay for L2 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr, weight_decay=alpha)

    early_escape_zero_rate = EarlyEscapeZeroRate(100)

    metric_list = []

    t = time.time()
    with trange(epochs) as titer:
        for e in titer:
            metric = {}
            total_loss = 0
            for x_batch, y_batch in dataloader:
                y_pred = model(x_batch)
                weight_dict = model.get_weights()
                l1_reg = 0
                loss = 0
                for k, w in weight_dict.items():
                    l1_reg += torch.norm(w, p=1)
                if loss_func == 'mse':
                    _func = torch.nn.MSELoss()
                    loss += _func(y_pred, y_batch)
                elif loss_func == 'ce':
                    _func = torch.nn.CrossEntropyLoss()
                    loss += _func(y_pred, y_batch)
                assert not torch.isnan(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() + alpha * l1_reg.item()

            epoch_loss = total_loss / len(dataloader)
            metric['epoch_loss'] = epoch_loss
            metric['epoch'] = e + 1
            metric['time'] = time.time() - t


            if (e + 1) % eval_every_epoch == 0:
                m = eval_over_linear_regression_datasets(x, y, model.get_weights(), alpha)
                metric.update(m)

                should_early_stop = True

                if epoch_loss < loss_less_than:
                    metric['loss_smaller_than_threshold'] = True
                    should_early_stop = should_early_stop and True
                else:
                    metric['loss_smaller_than_threshold'] = False
                    should_early_stop = should_early_stop and False

                for zrr in zero_rate_ratios:
                    if metric['zero_rate6'] >= zero_rate_greater_than * zrr:
                        metric[f'zero_rate_greater_than_threshold:{zrr}'] = True
                        should_early_stop = should_early_stop and True
                    else:
                        metric[f'zero_rate_greater_than_threshold:{zrr}'] = False
                        should_early_stop = should_early_stop and False

                metric_list.append(metric)

                if should_early_stop:
                    break

                if early_escape_zero_rate.check_escape(metric['zero_rate12']):
                    break

            titer.set_postfix(metric)

    t = time.time() - t

    weights = model.get_weights().reshape([predictor_dim, respond_dim])

    return {'time': t,
            'weights': weights,
            'metric_list': metric_list}


def run_classification(alpha, X_train, y_train, X_test, y_test,
                       net=None,
                       optname='SGD',
                       epochs=10000,
                       batch_size=1024,
                       lr=1e-4,
                       device='cuda:0',
                       eval_every_epoch=100, **kwargs):

    X_train_arr, X_valid_arr, y_train_arr, y_valid_arr = train_test_split(
        X_train, y_train, shuffle=True, test_size=0.2
    )

    X_train_ten = torch.tensor(X_train_arr, dtype=torch.float32, device=device)
    y_train_ten = torch.tensor(y_train_arr, dtype=torch.int64, device=device)
    X_valid_ten = torch.tensor(X_valid_arr, dtype=torch.float32, device=device)
    y_valid_ten = torch.tensor(y_valid_arr, dtype=torch.int64, device=device)

    # prepare the test set
    X_test_ten = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_ten = torch.tensor(y_test, dtype=torch.int64, device=device)


    dataset = torch.utils.data.TensorDataset(X_train_ten, y_train_ten)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = net

    # using weight decay for L2 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr, weight_decay=alpha)

    _func = torch.nn.CrossEntropyLoss()
    metric_list = []

    best_valid_acc = 0
    final_test_acc = 0

    t = time.time()
    with trange(epochs) as titer:
        for e in titer:
            model.train()
            metric = {}
            total_loss = 0
            total_ce = 0
            total_l1_reg = 0
            for X_batch, y_batch in dataloader:
                y_pred = model(X_batch)
                train_acc = (y_pred.argmax(-1) == y_batch).float().mean().item()
                loss = _func(y_pred, y_batch)
                assert torch.isfinite(loss)
                l1_reg = 0
                weight_dict = model.get_weights()
                for k, w in weight_dict.items():
                    l1_reg += torch.norm(w, p=1)
                assert torch.isfinite(l1_reg)
                total_ce += loss
                total_l1_reg += l1_reg
                total_loss += loss + alpha * l1_reg
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

            metric['cross_entropy'] = total_ce.item() / len(dataloader)
            metric['train_acc'] = train_acc
            metric['l1_reg'] = total_l1_reg.item() / len(dataloader)
            metric['epoch_loss'] = total_loss.item() / len(dataloader)
            metric['epoch'] = e + 1
            metric['time'] = time.time() - t

            if (e + 1) % eval_every_epoch == 0:
                model.eval()
                y_pred_valid = model(X_valid_ten).argmax(-1)
                valid_acc = (y_pred_valid == y_valid_ten).float().mean().item()
                y_pred_test = model(X_test_ten).argmax(-1)
                test_acc = (y_pred_test == y_test_ten).float().mean().item()
                metric['valid_acc'] = valid_acc
                metric['test_acc'] = test_acc

                total_numel = 0
                non_zero_numel = 0
                weight_dict = model.get_weights()
                for k, w in weight_dict.items():
                    total_numel += w.numel()
                    non_zero_numel += (torch.abs(w) > 1e-20).float().sum()
                metric['compression'] = (total_numel / non_zero_numel).item()

                titer.set_postfix(metric)
                metric_list.append(metric)
                logging.info(f"trajectory {metric}")

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    final_test_acc = test_acc


    return final_test_acc, metric_list



def run_sparse_feature_classification(alpha, X_train, y_train, X_test, y_test,
                        net=None,
                        optname='SGD',
                        epochs=10000,
                        batch_size=1024,
                        lr=1e-4,
                        device='cuda:0',
                        eval_every_epoch=100, **kwargs):

    X_train_arr, X_valid_arr, y_train_arr, y_valid_arr = train_test_split(
        X_train, y_train, shuffle=True, test_size=0.2
    )

    X_train_ten = torch.tensor(X_train_arr, dtype=torch.float32, device=device)
    y_train_ten = torch.tensor(y_train_arr, dtype=torch.int64, device=device)
    X_valid_ten = torch.tensor(X_valid_arr, dtype=torch.float32, device=device)
    y_valid_ten = torch.tensor(y_valid_arr, dtype=torch.int64, device=device)

    # prepare the test set
    X_test_ten = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_ten = torch.tensor(y_test, dtype=torch.int64, device=device)


    dataset = torch.utils.data.TensorDataset(X_train_ten, y_train_ten)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = net

    # using weight decay for L2 regularization
    model.to(device)
    if isinstance(model, SparseFeatureNet):
        optimizer_in = getattr(torch.optim, optname)(
            model.input.parameters(), lr=lr, weight_decay=alpha)

        optimizer_out = getattr(torch.optim, optname)(
            model.output.parameters(), lr=lr, weight_decay=1e-5)
    elif isinstance(model, SparseFeatureNetv2):
        optimizer_in = getattr(torch.optim, optname)(
            [{'params': model.input_mask},
            {'params': model.linear_output.parameters()},
            {'params': model.linear_feature.parameters()}], lr=lr, weight_decay=alpha)

        optimizer_out = torch.optim.Adam(
            [{'params': model.mlp_output.parameters()}], lr=4e-3, weight_decay=1e-5)
    else:
        optimizer_in = getattr(torch.optim, optname)(
             model.parameters(), lr=lr, weight_decay=alpha)
        optimizer_out = None


    _func = torch.nn.CrossEntropyLoss()
    metric_list = []

    best_valid_acc = 0
    final_test_acc = 0
    final_features = 0

    t = time.time()
    with trange(epochs) as titer:
        for e in titer:
            metric = {}
            total_loss = 0
            total_ce = 0
            total_l1_reg = 0
            for X_batch, y_batch in dataloader:
                y_linear_pred, y_pred = model(X_batch)
                train_acc = (y_pred.argmax(-1) == y_batch).float().mean().item()
                loss = _func(y_pred, y_batch)
                loss += _func(y_linear_pred, y_batch)
                loss /= 2
                assert torch.isfinite(loss)
                l1_reg = 0
                weight_dict = model.get_weights()
                for k, w in weight_dict.items():
                    l1_reg += torch.norm(w, p=1)
                assert torch.isfinite(l1_reg)
                total_ce += loss
                total_l1_reg += l1_reg
                total_loss += loss + alpha * l1_reg
                optimizer_in.zero_grad()
                if optimizer_out: optimizer_out.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer_in.step()
                if optimizer_out: optimizer_out.step()

            metric['cross_entropy'] = total_ce.item() / len(dataloader)
            metric['train_acc'] = train_acc
            metric['l1_reg'] = total_l1_reg.item() / len(dataloader)
            metric['epoch_loss'] = total_loss.item() / len(dataloader)
            metric['epoch'] = e + 1
            metric['time'] = time.time() - t

            if (e + 1) % eval_every_epoch == 0:
                model.eval()
                y_pred_valid = model(X_valid_ten)[1].argmax(-1)
                valid_acc = (y_pred_valid == y_valid_ten).float().mean().item()
                y_pred_test = model(X_test_ten)[1].argmax(-1)
                metric['#labels'] = len(Counter(y_pred_test.cpu().numpy().tolist()))
                test_acc = (y_pred_test == y_test_ten).float().mean().item()
                metric['valid_acc'] = valid_acc
                metric['test_acc'] = test_acc

                num_feat_select = (torch.abs(model.input_mask) > 1e-10).float().sum()
                metric['#features'] = num_feat_select.item()

                titer.set_postfix(metric)
                metric_list.append(metric)
                logging.info(f"trajectory {metric}")

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    final_test_acc = test_acc
                    final_features = num_feat_select

    return final_test_acc, final_features, metric_list

def run_l1_regression(alpha, x, y,
                      optname='SGD',
                      epoch=200,
                      batch_size=512,
                      lr=1e-4,
                      device='cuda:0',
                      loss_requirement=0,
                      eval_every_epoch=True):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = LinearRegression(input_dim=predictor_dim,
                             output_dim=respond_dim)

    # using L1 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr)

    early_escape_zero_rate = EarlyEscapeZeroRate(100)

    metric_list = []

    t = time.time()
    for e in range(epoch):
        metric = {}
        total_loss = 0
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            l1reg = model.L1_reg()
            loss = torch.sum((y_batch - y_pred) ** 2) / 2 / \
                   y_pred.size(0) + alpha * l1reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        metric['epoch_loss'] = epoch_loss
        metric['epoch'] = e + 1
        if eval_every_epoch:
            m = eval_over_linear_regression_datasets(x, y, model.get_weights(), alpha)
            metric.update(m)
            if early_escape_zero_rate.check_escape(metric['zero_rate12']):
                break

        metric_list.append(metric)

        if epoch_loss < loss_requirement:
            break

    t = time.time() - t

    weights = model.get_weights().reshape((predictor_dim, respond_dim))

    return {'time': t,
            'weights': weights,
            'metric_list': metric_list}
