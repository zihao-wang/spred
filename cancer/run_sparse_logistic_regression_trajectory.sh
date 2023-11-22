for dataset in mnist 20news
do
    for alpha in 1e-3 1e-2 1e-1 1 10 100 1000
    do
        python3 logistic_regression_trajectory.py --dataset=$dataset --alpha=$alpha
    done
done