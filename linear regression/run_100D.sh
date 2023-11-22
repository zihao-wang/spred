PREDICTOR_DIM=100
RESPOND_DIM=1
echo $RESPOND_DIM $PREDICTOR_DIM

for optname in Adam SGD
do
    for lr in 1e-3 1e-2 1e-1
    do
        for reg in l1
        do
            python3 sparse_linear_regression.py \
                --device=cuda:1 \
                --regularization=$reg \
                --optname=$optname \
                --lr=$lr \
                --predictor_dim=$PREDICTOR_DIM \
                --respond_dim=$RESPOND_DIM \
                --output_folder=output/LinearRegression${PREDICTOR_DIM}_${RESPOND_DIM}/${reg}_${optname}_lr=${lr}
        done
    done
done