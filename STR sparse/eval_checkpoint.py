import torch
import os
import time

from torchvision import datasets, transforms

from model_eval.resnet_eval import ResNet18

from utils.logging import AverageMeter
from utils.eval_utils import accuracy
import tqdm


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str)
parser.add_argument("--thr", type=str, default="1e-5,1e-4,1e-3,1e-2,1e-1")
parser.add_argument("--data", type=int, default=10)
parser.add_argument("--method", type=str, default="VanillaConv")
parser.add_argument("--output_csv_name", type=str, default="out.csv")


class CIFAR10:
    def __init__(self):

        data_root = os.path.join("rawdata", "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": 20, "pin_memory": True} if use_cuda else {}


        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.CIFAR10(
            data_root,
            True,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                data_root,
                False,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
                download=True
            ),
            batch_size=256,
            shuffle=False,
            **kwargs
        )


class CIFAR100:
    def __init__(self):

        data_root = os.path.join("rawdata", "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": 20, "pin_memory": True} if use_cuda else {}


        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.CIFAR100(
            data_root,
            True,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                data_root,
                False,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=256,
            shuffle=False,
            **kwargs
        )



def evaluate_checkpoint(ckpt_path="runs/resnet18-l1-cifar10/l1=1e-3/prune_rate=0.0/0/checkpoints/model_best.pth",
                        dataset="cifar10",
                        conv_type="vanilla",
                        thr=1e-3,
                        device="cuda:0",
                        num_classes=100):

    model = ResNet18(conv_type=conv_type,num_classes=num_classes)
    state_dict = torch.load(ckpt_path)
    model_state_dict = {}

    for k, v in state_dict['state_dict'].items():
        model_state_dict[k[7:]] = v

    model.load_state_dict(model_state_dict)

    for n, m in model.named_modules():
        if hasattr(m, 'thr'):
            m.thr = thr
            print("setting threshold", n)

    val_loader = dataset.val_loader

    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    # switch to evaluate mode
    model.eval()

    model.to(device)

    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            images = images.to(device)
            target = target.to(device).long()

            # compute output
            output = model(images)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    acc = top1.avg

    nonzero_sum, total_sum = 0, 0
    for n, m in model.named_modules():
        if hasattr(m, 'getSparsity'):
            nonzero, total, _=  m.getSparsity(thr)
            nonzero_sum += nonzero
            total_sum += total
    compression_ratio = total_sum / nonzero_sum

    # return {"acc": acc, "cr": compression_ratio.item()}
    return compression_ratio.item(), acc




# evaluate_checkpoint(
# ckpt_path = "runs/resnet18-spared-cifar100/weight_decay=3e-4/prune_rate=0.0/checkpoints/model_best.pth",
# dataset = "cifar100",
# conv_type = "VanillaConv1",
# thr = 1e-3,
# device="cuda:0",
# num_classes=100
# )


def identify_ckpt_in_sub_folder(path):
    print("in", path)
    if path.endswith("model_best.pth"):
        ti_m = os.path.getmtime(path)
        return [(path, time.ctime(ti_m))]

    elif os.path.isdir(path):
        ret = []
        for _p in os.listdir(path):
            p = os.path.join(path, _p)
            r = identify_ckpt_in_sub_folder(p)
            if r:
                ret.extend(r)
        return ret

    else:
        return []


if __name__ == "__main__":

    
    args = parser.parse_args()
    dataset = f"cifar{args.data}"
    if args.data == 10:
        dataset = CIFAR10()
    else:
        dataset = CIFAR100()

    ckpt_path_time_list = identify_ckpt_in_sub_folder(args.ckpt)

    print(ckpt_path_time_list)
    
    from collections import defaultdict
    data = defaultdict(list)

    
    for ckpt_path, ckpt_time in ckpt_path_time_list:

        for thr_str in args.thr.split(','):
            thr = float(thr_str)

            cr, acc = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                dataset=dataset,
                conv_type=args.method,
                thr=thr,
                device="cuda:0",
                num_classes=args.data
            )

            data['cr'].append(cr)
            data['acc'].append(acc)
            data['ckpt_path'].append(ckpt_path)
            data['ckpt_time'].append(ckpt_time)

    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(args.output_csv_name, index=False)
    print(df.to_string())
