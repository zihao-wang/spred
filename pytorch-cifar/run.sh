dataset=$1
pretrain_wd=$2
device=$3

pretrain_output_path=pretrain/${dataset}_pretrain_wd_${pretrain_wd}

python pretrain.py --dataset=$dataset --wd=$pretrain_wd --device=$device\
				   --output_path=$pretrain_output_path

for thr in 5e-1 1e-1 8e-2 6e-2 4e-2 2e-2 1e-2 1e-3 1e-4 1e-5 1e-6
do
	python3 finetune.py --lr=1e-4 --threshold=$thr --device=$device \
						--dataset=$dataset \
						--resume_path $pretrain_output_path \
						--output_path finetune/${dataset}_pretrain_wd_${pretrain_wd}/thr_${thr}
done
