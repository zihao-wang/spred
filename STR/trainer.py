import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):

    if args.threshold > 0:
        for n, m in model.named_modules():
            if hasattr(m, "set_spred_threshold"):
                print("setting the threshold of ", n)
                m.set_spred_threshold(args.threshold)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True).long()

        # compute output
        output = model(images)

        loss = criterion(output, target.view(-1))

        if args.l1_reg > 0:
            l1_reg = 0
            for p in model.parameters():
                l1_reg += p.abs().sum()
            loss = loss + l1_reg * args.l1_reg

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)
        #     with torch.no_grad():
        #     #    total = 0
        #        total_sparse = 0
        #        for parameter in model.parameters():
        #            total = torch.tensor(parameter.shape).prod() + total
        #            total_sparse = (parameter.abs() < 1e-2).sum() + total_sparse



        #     print(total_sparse/total)
        # if True and i>100:
        #    break

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch, prefix="Test:"):
    if args.threshold > 0:
        for n, m in model.named_modules():
            if hasattr(m, "set_spred_threshold"):
                print("setting the threshold of ", n)
                m.set_spred_threshold(args.threshold)

    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], f"{prefix}: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix, global_step=epoch)

    return top1.avg, top5.avg

def validate_with_sparse_ratio(val_loader, model, criterion, args, writer, epoch):
    # todo, bypass the sparse evaluation at this stage
    from copy import deepcopy
    import numpy as np

    records = []

    ratios = np.array(['0.0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.25','2.5','2.75','3.0','3.25','3.5','3.75','4.0','4.25','4.5','4.75','5.0','5.25','5.5','5.75','6.0'])
    x_ratios = np.linspace(0,6,len(ratios))
    for cr in x_ratios:
        sparse_model = deepcopy(model)
        sparse_ratio = 1 - 10 ** (-cr)
        for n, m in sparse_model.named_modules():
            if hasattr(m, 'setSparseRatio'):
                print("prune", n, "under sparse ratio", sparse_ratio)
                m.setSparseRatio(sparse_ratio)
                m.prune()

        top1acc, top5acc = validate(
            val_loader, sparse_model, criterion, args, writer, epoch, prefix=f"Test@CR={cr}")
        records.append(
            (sparse_ratio, top1acc, top5acc)
        )
