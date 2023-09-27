import os
import tqdm
import random
from PIL import Image
import numpy as np
import pandas as pd
import tqdm as tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data.sampler import SequentialSampler, RandomSampler
from sklearn.model_selection import StratifiedKFold

from efficientnet_b6 import *
from printing import progress_bar

data_path = os.path.abspath(os.path.join(os.path.curdir, 'input/cassava-leaf-disease-classification'))
train_path = os.path.join(data_path, 'train_images')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test_images')
csv_path = os.path.join(data_path, 'sample_submission.csv')
cwd = os.path.abspath('.')

cfg = {
    'fold_num': 5,
    'seed': 899,
    'model_arch': 'efficientnet-b6',
    'image_size': 512,
    'epochs': 8,
    'train_batch_size': 2,
    'val_batch_size': 8,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-7,
    'weight_decay': 1e-6,
    'num_workers': 1,
    'device': 'cuda:0'
}

# 旋转 空白像素用黑色填充fill
# 翻转
# 缩放 zoom in/out 20%
# 裁剪 不能把目标裁掉
# 平移 同样的
# 拉伸、收缩
# 可以把这个仿射变换综合在一起
# 增加噪声 - 高斯噪声
# 模糊化 - 高斯模糊
# RGB 颜色扰动
# 随机擦除、遮挡
dataset_transforms = {
    'train': transforms.Compose([
                    transforms.Resize((528, 528)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.RandomCrop((528, 528), padding=3),
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4304, 0.4968, 0.3135], [0.2358, 0.2387, 0.2256])
                ]),
    'other': transforms.Compose([
                    transforms.Resize((528, 528)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4304, 0.4968, 0.3135], [0.2358, 0.2387, 0.2256])
                ])
}

device = cfg['device']
best_acc = .0
label_df = pd.read_csv(label_path)

# 也许weighted cross entropy loss 反而不好
freq_ = dict(len(label_df) / label_df['label'].value_counts())
weights = torch.zeros(5)
for k, v in freq_.items():
    weights[k] = v
weights = weights.to(device)

cudnn.benchmark = True

class CassavaDataset(Dataset):
    def __init__(self, df, dPath=None, transforms=None, test=False):
        super(CassavaDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.dPath = dPath
        self.transforms = transforms
        self.test = test
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_name = self.df['image_id'].values[idx]
        img_path = os.path.join(self.dPath, img_name)
        img = Image.open(img_path)
        if self.test:
            label = int(img_name.split('.')[0])
        else:
            img = self.transforms(img)
            label = self.df['label'].values[idx]
        return img, label

def prepare_dataloader(label_df, train_idx, val_idx):
    train_df = label_df.loc[train_idx, :].reset_index(drop=True)
    val_df = label_df.loc[val_idx, :].reset_index(drop=True)
    
    train_dataset = CassavaDataset(train_df, train_path, dataset_transforms['train'])
    val_dataset = CassavaDataset(val_df, train_path, dataset_transforms['other'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    
    return train_loader, val_loader

def train(train_loader, net, optimizer, scheduler, criterion):
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

def validate(epoch, val_loader, net, optimizer, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.join(cwd, 'checkpoint')):
            os.mkdir(os.path.join(cwd, 'checkpoint'))
        torch.save(state, os.path.join(cwd, 'checkpoint/leaf-b6.pth'))
        best_acc = acc

def main_loop():
    folds = StratifiedKFold(n_splits=cfg['fold_num'], shuffle=True, random_state=cfg['seed'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df.label.values)
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f'{fold}th fold training starts...')
        
        train_loader, val_loader = prepare_dataloader(label_df, train_idx, val_idx)
        net = EfficientNet.from_pretrained('efficientnet-b6').to(device)
        net._fc = nn.Linear(net._fc.in_features, 5)
        net = torch.nn.DataParallel(net)
        
        optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=1, eta_min=cfg['min_lr'], last_epoch=-1)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        for epoch in range(cfg['epochs']):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(epoch, val_loader, net, optimizer, criterion)

main_loop()