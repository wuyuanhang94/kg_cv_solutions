import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import 
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
from dataset import MyDataset

parser = argparse.ArgumentParser(description='Cat or Dog')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = .0
start_epoch = 0

# Data
print("==> Preparing data...")

full_dataset = MyDataset(data_path='./input/train', train=True)
train_size = int(0.8 * len(full_dataset))
validate_size = int(0.2 * len(full_dataset))

train_set, validate_set = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=2, num_workers=4)

# Model
print("==> Building model...")
net = models.resnet50(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

net.fc = nn.Linear(2048, 2)
net = net.to(device)
print(net)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint
    print("==> Resuming from checkpoint...")
    assert os.path.isdir('checkpoint'), "Error: no checkpoint directory found"
    checkpoint = torch.load('./checkpoint/dog-cat-resnet50-pretrained.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def validate(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(validate_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar(batch_idx, len(validate_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/dog-cat-resnet50-pretrained.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    validate(epoch)
