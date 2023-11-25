import GEOparse
import numpy as np
import logging

def get_cancer_GDS(filepath):
    gds = GEOparse.get_GEO(filepath=filepath)
    X = []
    y = []
    subset_keys = list(gds.subsets.keys())
    for i, k in enumerate(subset_keys):
        sample_ids = gds.subsets[k].metadata['sample_id'][0].split(',')
        for sample_id in sample_ids:
            _x = gds.table.loc[:, sample_id].to_numpy().reshape((1, -1))
            _y = i
            X.append(_x)
            y.append(_y)

    from collections import Counter
    yc = Counter(y)
    for k, c in yc.most_common(1):
        logging.info(f"most common label {k}: percent {c / len(y)}")

    X_arr = np.concatenate(X, axis=0).astype('float32')
    X_arr = np.nan_to_num(X_arr, nan=0)
    y_arr = np.asarray(y).astype('int64')

    X_arr_mean = np.mean(X_arr, axis=0)
    X_arr_std = np.std(X_arr)
    X_arr -= X_arr_mean
    X_arr /= (X_arr_std + 1e-10)

    logging.info(f"GDS dataset {filepath} loaded")
    logging.info(f"#features {X_arr.shape[1]}, #labels {np.max(y_arr)+1}, #samples {X_arr.shape[0]}")
    return X_arr, y_arr
