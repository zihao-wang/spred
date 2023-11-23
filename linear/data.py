import numpy as np
from sklearn.model_selection import train_test_split


def isotropic_predictor_data(num_samples, predictor_dim, respond_dim, noisy_variance, sparse=0, seed=666):
    np.random.seed(seed)
    x = np.random.randn(num_samples, predictor_dim)

    trans = np.random.randn(predictor_dim, respond_dim)
    if sparse > 0:
        sparse_mask = np.random.rand(trans.shape)
        trans = (sparse_mask > sparse) * trans

    y = x.dot(trans) + np.random.randn(num_samples,
                                       respond_dim) * noisy_variance

    return (x, y), trans

def isotropic_predictor_data_torch(num_samples, predictor_dim, respond_dim, noisy_variance, sparse=0, seed=666):
    np.random.seed(seed)
    x = np.random.randn(num_samples, predictor_dim)

    trans = np.random.randn(predictor_dim, respond_dim)
    if sparse > 0:
        sparse_mask = np.random.rand(trans.shape)
        trans = (sparse_mask > sparse) * trans

    y = x.dot(trans) + np.random.randn(num_samples,
                                       respond_dim) * noisy_variance

    return (x, y), trans
