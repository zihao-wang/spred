for weight_decay in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2
do
    for threshold in 1e-6 1e-5 1e-4 1e-3 1e-2
    do
    python3 main.py --config configs/largescale/resnet18-spred-cifar100.yaml \
                    --weight-decay $weight_decay \
                    --name pretrain_weight_decay=$weight_decay-finetune_threshold=$threshold \
                    --multigpu 1 \
                    --num_classes 100 \
                    --pretrained runs/resnet18-spred-cifar100/pretrain_weight_decay=$weight_decay/checkpoints/model_best.pth \
                    --threshold $threshold
    done
done