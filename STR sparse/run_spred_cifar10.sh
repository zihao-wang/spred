for weight_decay in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2
do
    python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
                    --weight-decay $weight_decay \
                    --name pretrain_weight_decay=$weight_decay \
                    --multigpu 0
done