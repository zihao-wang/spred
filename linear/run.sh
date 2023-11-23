rm -rf output/LinearRegression
predictor_dim=$1
respond_dim=$2

echo "predictor_dim=$predictor_dim"
echo "respond_dim=$respond_dim"

for optname in Adam SGD
do
    for lr in 1 1e-1 1e-2
    do
        python3 high_dim_regression.py \
            --device=cuda:0 \
            --optname=$optname \
            --lr=$lr \
            --predictor_dim=$predictor_dim \
            --respond_dim=$respond_dim \
            --num_alpha=10
    done
done