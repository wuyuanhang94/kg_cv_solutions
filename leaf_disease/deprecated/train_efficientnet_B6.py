import os
import cv2
import random
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

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

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

data_path = os.path.abspath(os.path.join(os.path.curdir, 'input/cassava-leaf-disease-classification'))
train_path = os.path.join(data_path, 'train_images')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test_images')
csv_path = os.path.join(data_path, 'sample_submission.csv')
cwd = os.path.abspath('.')

cfg = {
    'fold_num': 5,
    'seed': 899,
    'model_arch': 'efficientnet-b4',
    'img_size': 512,
    'epochs': 20,
    'train_batch_size': 16,
    'val_batch_size': 16,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(img_path):
    img_bgr = cv2.imread(img_path)
    return img_bgr[..., ::-1]

def get_train_transforms():
    return Compose([
            RandomResizedCrop(cfg['img_size'], cfg['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0)
        ], p=1.0)

def get_valid_transforms():
    return Compose([
            CenterCrop(cfg['img_size'], cfg['img_size'], p=1.),
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

device = cfg['device']
best_acc = .0
label_df = pd.read_csv(label_path)
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
        img = get_img(img_path)
        if self.test:
            img = np.array(img)
            label = int(img_name.split('.')[0])
        else:
            img = self.transforms(image=img)['image']
            label = self.df['label'].values[idx]
        return img, label

def prepare_dataloader(label_df, train_idx, val_idx):
    train_df = label_df.loc[train_idx, :].reset_index(drop=True)
    val_df = label_df.loc[val_idx, :].reset_index(drop=True)
    
    train_dataset = CassavaDataset(train_df, train_path, get_train_transforms())
    val_dataset = CassavaDataset(val_df, train_path, get_valid_transforms())
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    
    return train_loader, val_loader

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, labels) in progress_bar:
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

        description = 'Loss: %.3f | Acc: %.3f%%' % (train_loss/(batch_idx+1), 100.*correct/total)
        progress_bar.set_description(description)

def validate(val_loader, net, optimizer, criterion, fold):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            description = 'Loss: %.3f | Acc: %.3f%%' % (test_loss/(batch_idx+1), 100.*correct/total)
            progress_bar.set_description(description)

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc
        }
        if not os.path.isdir(os.path.join(cwd, 'checkpoint')):
            os.mkdir(os.path.join(cwd, 'checkpoint'))
        torch.save(state, os.path.join(cwd, f'checkpoint/leaf-b4-fold{fold}.pth'))
        best_acc = acc

def main_loop():
    torch.cuda.empty_cache()
    seed_everything(cfg['seed'])

    folds = StratifiedKFold(n_splits=cfg['fold_num'], shuffle=True, random_state=cfg['seed'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df.label.values)
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        global best_acc
        best_acc = 0.
        print(f'{fold}th fold training starts...')

        train_loader, val_loader = prepare_dataloader(label_df, train_idx, val_idx)
        net = EfficientNet.from_pretrained('efficientnet-b4').to(device)
        net._fc = nn.Linear(net._fc.in_features, 5)
        net = torch.nn.DataParallel(net)

        optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=1, eta_min=cfg['min_lr'], last_epoch=-1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(cfg['epochs']):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(val_loader, net, optimizer, criterion, fold)

main_loop()