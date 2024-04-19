import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--variant', default="default", type=str, help='model activation function to use (LIF, ParaLIF)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='SimpleDLA', type=str, help='model to use (e.g., VGG19, ResNet18)')
args = parser.parse_args()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f'==> Using {device} for backend')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_dict = {
    'VGG11': VGG('VGG11'),
    'VGG13': VGG('VGG13'),
    'VGG16': VGG('VGG16'),
    'VGG19': VGG('VGG19'),
    'ResNet18': ResNet18(),
    'ResNet34': ResNet34(),
    'ResNet101': ResNet101(),
    'ResNet152': ResNet152(),
    'GoogLeNet': GoogLeNet(),
    'DenseNet121': DenseNet121(),
    'DenseNet161': DenseNet161(),
    'DenseNet169': DenseNet169(),
    'DenseNet201': DenseNet201(),
    'MobileNet': MobileNet(),
    'EfficientNetB0': EfficientNetB0(),
    'ResNeXt29_2x64d': ResNeXt29_2x64d(),
    'PreActResNet18': PreActResNet18(),
    'SimpleDLA': SimpleDLA(),
    'DLA': DLA(),
    'LeNet': {"default": LeNet(), "LIF": LeNet_LIF()},
    'LeNet5': {"default": LeNet5(), "LIF": LeNet5_LIF()},
}
if args.model not in model_dict:
    raise ValueError(f"Model {args.model} not recognized. Available models: {list(model_dict.keys())}")

model_entry = model_dict[args.model]
if isinstance(model_entry, dict):
    if args.variant not in model_entry:
        raise ValueError(f"Variant {args.variant} not recognized for model {args.model}. Available variants: {list(model_entry.keys())}")
    net = model_entry[args.variant]
else:
    net = model_entry

net = net.to(device)

print(f'==> Building "{args.variant}" {args.model} model..')

if args.resume:
    checkpoint_path = f'./checkpoint/ckpt_{args.model}_{args.variant}.pth'
    if os.path.isfile(checkpoint_path):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        if args.variant == 'LIF':
            checkpoint['net'] = {k: v for k, v in checkpoint['net'].items() if 'mem' not in k}
            net.load_state_dict(checkpoint['net'], strict=False)
        else:
            net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr) # momentum=0.9, weight_decay=5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        checkpoint_dir = './checkpoint'
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{args.model}_{args.variant}.pth')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        torch.save(state, checkpoint_path)
        best_acc = acc
    else:
        print('Not improved, skipping save..')

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
