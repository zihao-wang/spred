'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.resnet import ResNet18
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--threshold', default=1e-2, type=float, help='pruning threshold')
parser.add_argument('--finetune_epochs', default=20, type=int, help='finetune epochs')

parser.add_argument("--dataset", default="CIFAR10", type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--output_path', type=str, default='finetune')
parser.add_argument('--resume_path', type=str)

args = parser.parse_args()

output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)

device = args.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset.lower() == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
elif args.dataset.lower() == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)

if args.dataset.lower() == 'cifar10':
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
elif args.dataset.lower() == 'cifar100':
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_train)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=500, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.dataset.lower() == 'cifar10':
    net = ResNet18(10)
elif args.dataset.lower() == 'cifar100':
    net = ResNet18(100)
net = net.to(device)


thresh = args.threshold
if args.resume_path:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(args.resume_path, 'last.pt'),
                            map_location=device)
    net_state_dict = checkpoint['net']
    net.load_state_dict(net_state_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    print(f"pruning by {thresh} before finetuning")
    total_param = 0.
    spar_param = 0.
    for parameter in net.parameters():
        total_param = total_param + torch.prod(torch.tensor(parameter.shape))
        spar_param = spar_param + (parameter.abs() < thresh).sum()
        parameter.data[parameter.abs() <thresh] = 0
    sparsity = spar_param / total_param
    print('current sparsity: ', sparsity)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr,
                       weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    total_param = 0.
    spar_param = 0.
    for parameter in net.parameters():
        total_param = total_param + torch.prod(torch.tensor(parameter.shape))
        spar_param = spar_param + (parameter.abs() < thresh).sum()
        #parameter.data[parameter.abs() < thresh] = 0

    sparsity = spar_param / total_param


    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Comp:  %.4f'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 1/(1- sparsity)))

    # Save checkpoint.
    print("saving ...")
    acc = 100.*correct/total
    state = {
        'acc': acc,
        'epoch': epoch,
        'net': net.state_dict(),
    }
    torch.save(state, os.path.join(output_dir, 'last.pt'))
    if acc > best_acc:
        print('Saving..')
        torch.save(state,
            os.path.join(output_dir, 'best.pt')
            )
        best_acc = acc

for epoch in range(start_epoch, start_epoch+args.finetune_epochs):
    train(epoch)
    test(epoch)
