# model_name=sparse_feature_net
# do
#     for alpha in 1e-2 5e-3 2e-3 1e-3 9e-4
#     do
#         for optname in Adam SGD
#         do
#             python3 cancer_sparse_classification.py \
#                 --model_name=$model_name --alpha=$alpha --optname=$optname --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr1e-3 \
#                 --lr=1e-3 --device="cuda:0" &
#             python3 cancer_sparse_classification.py \
#                 --model_name=$model_name --alpha=$alpha --optname=$optname --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr1e-4 \
#                 --lr=1e-4 --device="cuda:1" &
#             wait
#         done
#     done
# done


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

# optname=SGD
# model_name=sparse_feature_net_v2
# # for alphacuda in "4e-1 cuda:0" "3e-1 cuda:1" "2e-2 cuda:2" "1e-2 cuda:3"
# for alpha in 7 5 3 1
# do
#     # for alphacuda in "3e-1 cuda:0" "1e-1 cuda:1" "3e-2 cuda:2" "1e-2 cuda:3"
#     # for lr in 1e-1 5e-2 1e-2 5e-3
#     for lr_cuda in "5e-2 cuda:0" "3e-2 cuda:1" "1e-2 cuda:2" "5e-3 cuda:3"
#     do
#         set -- $lr_cuda
#         echo ${1} and ${2}
#         python3 cancer_sparse_classification.py \
#             --model_name=$model_name \
#             --alpha=$alpha \
#             --optname=$optname \
#             --lr=$1 \
#             --device=$2 \
#             --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr${1}.log &
#     done
#     wait
# done
# #

# python3 cancer_sparse_classification.py --model_name sparse_feature_linear_svm --alpha 0.3 --lr 1e-1 --optname SGD --log_dir output/cancer/sparse_feature_linear_svm.log --device cuda:0 &
# python3 cancer_sparse_classification.py --model_name sparse_feature_linear --alpha 0.3 --lr 1e-1 --optname SGD --log_dir output/cancer/sparse_feature_linear.log --device cuda:1 &
# python3 cancer_sparse_classification.py --model_name mlp --alpha 1e-4 --lr 1e-4 --optname Adam --log_dir output/cancer/mlp.log --device cuda:2 &