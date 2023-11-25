# Run the experiment on all cancer classification datasets

To run our method:

python3 cancer_sparse_classification.py --model_name {test method name}

test method name includes
- logistic_regression
- hsiclasso
- sparse_feature_linear (This is the f_l in the paper)
- sparse_feature_linear_svm
- mlp
- sparse_feature_net
- sparse_feature_net_v2 (This is the f_l + f_n in the paper)