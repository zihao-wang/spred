for l1 in 1e-5 3e-5 1e-4 3e-4 1e-3
do
    python3 main.py --config configs/largescale/resnet18-l1-cifar100.yaml \
                    --l1-reg $l1 \
                    --name l1=$l1 \
                    --num_classes 100\
                    --multigpu 1
done