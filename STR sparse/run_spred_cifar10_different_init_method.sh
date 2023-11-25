#!/bin/bash

weight_decay=1e-3
threshold=1e-3

# Define two arrays
first_array=("xavier_uniform_" "kaiming_uniform_" "kaiming_normal_")
second_array=("0" "1" "2")

train() {
# python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
#                             --weight-decay $weight_decay \
#                             --name pretrain_weight_decay=$weight_decay-init=$init_method-red=$init_red \
#                             --init_method $init_method \
#                             --init_red $init_red \
#                             --multigpu $cuda &
python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
                            --weight-decay $weight_decay \
                            --name pretrain_weight_decay=$weight_decay-init=$1-red=$2 \
                            --init_method $1 \
                            --init_red $2 \
                            --multigpu $3

# python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
#                         --weight-decay $weight_decay \
#                         --name pretrain_weight_decay=$weight_decay-init=$1-red=$1-finetune_threshold=$threshold \
#                         --init_method $1 \
#                         --init_red $2 \
#                         --pretrained runs/resnet18-spred-cifar10/pretrain_weight_decay=$weight_decay-init=$1-red=$2/checkpoints/model_best.pth \
#                         --threshold $threshold \
#                         --multigpu $3
}

# train "xavier_uniform_" "independent" "0" &
# train "kaiming_uniform_" "independent" "1" &
# train "kaiming_normal_" "independent" "2" &

wait

train "xavier_uniform_" "sqrt" "0" &
train "kaiming_uniform_" "sqrt" "1" &
train "kaiming_normal_" "sqrt" "2" &