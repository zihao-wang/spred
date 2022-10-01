project_folder=$(pwd)
echo $project_folder
ssh cat0 "cd $project_folder && sh run.sh cifar10 1e-3 cuda:0 > cifar10_pretrainwd_1e-3.log &"
ssh cat0 "cd $project_folder && sh run.sh cifar10 2e-3 cuda:1 > cifar10_pretrainwd_2e-3.log &"
ssh cat1 "cd $project_folder && sh run.sh cifar100 1e-3 cuda:0 > cifar100_pretrainwd_1e-3.log &"
ssh cat1 "cd $project_folder && sh run.sh cifar100 2e-3 cuda:1 > cifar100_pretrainwd_2e-3.log &"