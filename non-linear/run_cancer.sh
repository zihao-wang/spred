optname=SGD
model_name=sparse_feature_linear
for lr in 5e-1 3e-1 1e-1 5e-2 3e-2 1e-1
do
    for alphacuda in "3e-1 cuda:0" "1e-1 cuda:1" "3e-2 cuda:2" "1e-2 cuda:3"
    # for alphacuda in "6e-1 cuda:0" "7e-1 cuda:1" "8e-2 cuda:2" "9e-2 cuda:3"
    do
        set -- $alphacuda
        echo ${1} and $2
        python3 cancer_sparse_classification.py \
            --model_name=$model_name \
            --alpha=$1 \
            --optname=$optname \
            --lr=$lr \
            --device=$2 \
            --log_dir=output/cancer_${model_name}_alpha=${1}_${optname}_lr${lr}.log &
    done
    wait
done
