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

from sklearn.model_selection import StratifiedKFold

from efficientnet import *
from loss import *
import warnings
warnings.filterwarnings("ignore")
import argparse

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, ElasticTransform,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, GaussianBlur, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, GridDropout
)
from albumentations.pytorch import ToTensorV2

data_path = os.path.abspath(os.path.join(os.path.curdir, 'input/cassava-leaf-disease-classification'))
train_path = os.path.join(data_path, 'train_images')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test_images')
csv_path = os.path.join(data_path, 'sample_submission.csv')
cwd = os.path.abspath('.')
parser = argparse.ArgumentParser()
device_name = torch.cuda.get_device_name()
if device_name == 'TITAN V':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

cfg = {
    'fold_num': 5,
    'seed': 899,
    'model_arch': 'efficientnet-b4',
    'img_size': 456,
    'epochs': 15,
    'train_batch_size': 4 if device_name != 'TITAN V' else 16,
    'val_batch_size': 4, #if device_name != 'TITAN V' else 16,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'accum_iter': 4,
    'weight_decay': 1e-6,
    't1': 1.0,
    't2': 1.5,
    'label_smoothing': 0.12,
    'num_workers': 8,
    'device': 'cuda'
}

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
            Resize(cfg['img_size'], cfg['img_size']),
            RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.8, 1.0)),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            IAASharpen(p=0.5),
            IAAEmboss(p=0.5),
            CLAHE(clip_limit=4.0, p=0.5),
            ShiftScaleRotate(p=0.5),
            RandomRotate90(),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),

            OneOf([
                MotionBlur(blur_limit=5),
                MedianBlur(blur_limit=5),
                GaussianBlur(blur_limit=5),
                GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.3),

            OneOf([
                OpticalDistortion(distort_limit=0.8),
                GridDistortion(num_steps=5, distort_limit=0.8),
                ElasticTransform(alpha=3),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.8),

            Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            CoarseDropout(max_holes=40, max_height=16, max_width=16, p=0.7),
            Cutout(num_holes=1, max_h_size=int(0.2*cfg['img_size']), max_w_size=int(0.2*cfg['img_size']), p=0.5),
            GridDropout(p=0.2),
            ToTensorV2(p=1.0)
        ], p=1.0)

def get_valid_transforms():
    return Compose([
            Resize(cfg['img_size'], cfg['img_size']),
            CenterCrop(cfg['img_size'], cfg['img_size'], p=1.),
            Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

device = cfg['device']
best_acc = 0.
best_loss = 99.0
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
        img = self.transforms(image=img)['image']
        if self.test:
            label = int(img_name.split('.')[0])
        else:
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

def save_model(net, fold, mode='acc', acc=.0, loss=.0):
    global best_acc
    global best_loss

    state = {
        'net': net.state_dict(),
        'acc': acc,
        'loss': loss
    }
    if not os.path.isdir(os.path.join(cwd, 'checkpoint')):
        os.mkdir(os.path.join(cwd, 'checkpoint'))
    if mode == 'acc':
        print('Saving acc...')
        torch.save(state, os.path.join(cwd, f'checkpoint/leaf-b4-fold{fold}-acc.pth'))
        best_acc = acc
    else:
        torch.save(state, os.path.join(cwd, f'checkpoint/leaf-b4-fold{fold}-loss.pth'))
        best_loss = loss

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss /= cfg['accum_iter']
        loss.backward()
        if ((batch_idx+1) % cfg['accum_iter'] == 0 or batch_idx == len(train_loader)-1):
            optimizer.step()
            net.zero_grad()
            optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        description = 'Loss: %.3f | Acc: %.3f%%' % (train_loss/(batch_idx+1), 100.*correct/total)
        progress_bar.set_description(description)

def validate(val_loader, net, optimizer, criterion, fold):
    global best_acc
    global best_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            x = torch.stack([inputs,
                             transforms.RandomHorizontalFlip(p=1)(inputs),
                             transforms.RandomVerticalFlip(p=1)(inputs),
                             inputs.transpose(-1,-2),
                             transforms.RandomHorizontalFlip(p=1)(inputs.transpose(-1,-2)),
                             transforms.RandomVerticalFlip(p=1)(inputs.transpose(-1,-2)),
                             torch.rot90(inputs, k=1, dims=[-1,-2]),
                             torch.rot90(inputs, k=2, dims=[-1,-2]),
                             torch.rot90(inputs, k=3, dims=[-1,-2])
                ], 1)
            x = x.view(-1, 3, cfg['img_size'], cfg['img_size'])
            outputs = net(x)
            outputs = outputs.view(inputs.shape[0], 9, -1).mean(1)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            description = 'Loss: %.3f | Acc: %.3f%%' % (test_loss/(batch_idx+1), 100.*correct/total)
            progress_bar.set_description(description)

    acc = 100.*correct/total
    test_loss /= len(val_loader)
    if acc > best_acc:
        save_model(net, fold, 'acc', acc, test_loss)
    if test_loss < best_loss:
        save_model(net, fold, 'loss', acc, test_loss)

def main_loop(resume, adam):
    torch.cuda.empty_cache()
    seed_everything(cfg['seed'])

    folds = StratifiedKFold(n_splits=cfg['fold_num'], shuffle=True, random_state=cfg['seed'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df.label.values)
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        global best_acc
        global best_loss
        best_acc = 0.
        best_loss = 0.62
        print(f'\n{fold}th fold training starts...')

        train_loader, val_loader = prepare_dataloader(label_df, train_idx, val_idx)

        if resume:
            net = EfficientNet.from_name('efficientnet-b4').to(device)
            net._fc = nn.Linear(net._fc.in_features, 5)
            net = torch.nn.DataParallel(net, device_ids=[0, 1] if device_name == 'TITAN V' else [0])
            net = net.cuda()            
            checkpoint = torch.load(f'checkpoint/leaf-b4-fold{fold}-acc.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
        else:
            net = EfficientNet.from_pretrained('efficientnet-b4').to(device)
            net._fc = nn.Linear(net._fc.in_features, 5)
            net = torch.nn.DataParallel(net, device_ids=[0, 1] if device_name == 'TITAN V' else [0])
            net = net.cuda() 

        if adam:
            optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=1, eta_min=cfg['min_lr'], last_epoch=-1)
        criterion = BiTemperedLoss(t1=cfg['t1'], t2=cfg['t2'], label_smoothing=cfg['label_smoothing'])

        epochs = 6 if resume else cfg['epochs']
        for epoch in range(epochs):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(val_loader, net, optimizer, criterion, fold)

        del net, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=1)
parser.add_argument('--Adam', type=int, help='use Adam or SGD', default=1)
args = parser.parse_args()

main_loop(args.resume, args.Adam)
