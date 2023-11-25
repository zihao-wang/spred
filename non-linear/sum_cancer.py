import os
import json
import pandas as pd
from collections import defaultdict


if __name__ == "__main__":
    input_file_prefix = "output/cancer_sparse_feature_linear"
    folder = os.path.dirname(input_file_prefix)
    prefix = os.path.basename(input_file_prefix)
    data = defaultdict(list)
    for f in os.listdir(folder):
        if prefix in f:
            filename = os.path.join(folder, f)
            with open(filename, 'rt') as f:
                for line in f.readlines():
                    if 'metric:' in line:
                        _, metric = line.split('metric:')
                        d = eval(metric)
                        print(d)
                        for k in d:
                            data[k].append(d[k])
    df = pd.DataFrame(data)
    df.to_csv(f'output/{prefix}.csv', index=False)
