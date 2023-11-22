for weight_decay in 1e-5 3e-5 1e-4 3e-4 1e-3
do
    python3 main.py --config configs/largescale/resnet18-str-cifar100.yaml \
                    --weight-decay $weight_decay \
                    --name weight_decay=$weight_decay \
                    --num_classes 100 \
                    --multigpu 3
done