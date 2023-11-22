rm -rf fast_runs
fast_spred() {
    python3 main.py --config configs/largescale/resnet18-spred-cifar10.yaml \
                    --weight-decay $1 \
                    --name fast_run_wd=$1 \
                    --multigpu $2 \
                    --pretrained runs/resnet18-spred-cifar10/pretrain_weight_decay=1e-4/checkpoints/model_best.pth \
                    --log-dir fast_runs

}

fast_spred 1e-5 0 &
fast_spred 1e-4 1 &
fast_spred 1e-3 2 &
fast_spred 1e-2 3 &

wait

fast_spred 3e-5 0 &
fast_spred 3e-4 1 &
fast_spred 3e-3 2 &
fast_spred 3e-2 3 &

wait