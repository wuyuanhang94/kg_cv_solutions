import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import ast
import cv2
import random
from PIL import Image
import numpy as np
import pandas as pd
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import timm
import argparse
import warnings
warnings.filterwarnings("ignore")
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, ElasticTransform,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, GaussianBlur, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, GridDropout
)
from albumentations.pytorch import ToTensorV2

project_path = os.path.abspath('.')
data_path = os.path.join(project_path, 'input')
train_path = os.path.join(data_path, 'train')
label_path = os.path.join(data_path, 'train_level.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')

label_df = pd.read_csv("/raid/yiw/siim/train_df.csv")

parser = argparse.ArgumentParser()

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_arch': 'tf_efficientnetv2_l',
    'img_size': 640,
    'epochs': 2,
    'train_batch_size': 30,
    'val_batch_size': 32,
    'T_0': 2,
    'T_mul': 1,
    'lr': 1e-5,
    'min_lr': 8e-6,
    'accum_iter': 1,
    'weight_decay': 1e-6,
    'num_workers': 12,
    'device': 'cuda',
    'skip_fold': 0,
    'device_num': 8
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
            # Resize(cfg['img_size'], cfg['img_size']),
            RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),

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
            ], p=0.3),

            Cutout(num_holes=10, max_h_size=int(0.1*cfg['img_size']), max_w_size=int(0.1*cfg['img_size']), p=0.5),
            
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ], p=1.0)

def get_valid_transforms():
    return Compose([
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ], p=1.0)

device = cfg['device']

best_auc = 0.
target_cols = label_df.iloc[:, 4:8].columns.tolist()

class SiimDataset(Dataset):
    def __init__(self, df, transforms=None, test=False):
        super(SiimDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.path = df['path']
        self.transforms = transforms
        self.labels = df[target_cols].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.path[idx]
        image = get_img(img_path)
        augmented = self.transforms(image=image)
        img = augmented['image']
        
        labels = torch.tensor(self.labels[idx]).float()
        return img, labels

class CustomEffNet(nn.Module):
    def __init__(self, model_name=cfg['model_arch'], pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 4)

    def forward(self, x):
        return self.model(x)

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.09):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing 

    def forward(self, logits, labels):
        labels[labels == 1] = 1 - self.smoothing 
        labels[labels == 0] = self.smoothing 
        return F.binary_cross_entropy_with_logits(logits, labels)

def prepare_dataloader(train_idx, val_idx):
    train_df = label_df.loc[train_idx, :].reset_index(drop=True)
    val_df = label_df.loc[val_idx, :].reset_index(drop=True)

    train_dataset = SiimDataset(df=train_df, transforms=get_train_transforms())
    val_dataset = SiimDataset(df=val_df, transforms=get_valid_transforms())
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    return train_loader, train_df, val_loader, val_df

def save_model(net, fold, auc=.0, loss=.0):
    state = {
        'net': net.state_dict(),
        'auc': auc,
        'loss': loss
    }
    print(f'Saving auc: {auc}...')
    model_arch = cfg['model_arch']
    torch.save(state, os.path.join(project_path, f'checkpoint/siim-study-{model_arch}-fold{fold}-640.pth'))

def get_auc_score(labels, outputs):
    aucs = []
    for j in range(labels.shape[1]):
        aucs.append(roc_auc_score(labels[:, j], outputs[:, j]))
    avg_auc = np.mean(aucs)
    return avg_auc, aucs

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0.
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss /= cfg['accum_iter']
        loss.backward()
        if ((batch_idx+1) % cfg['accum_iter'] == 0 or batch_idx == len(train_loader)-1):
            optimizer.step()
            net.zero_grad()
            optimizer.zero_grad()

        train_loss += loss.item()
        description = 'Loss: %.3f' % (train_loss/(batch_idx+1))
        progress_bar.set_description(description)

    scheduler.step()

def validate(val_loader, net, criterion, fold, val_labels):
    global best_auc
    net.eval()
    test_loss = 0.
    preds = []

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    for batch_idx, (inputs, labels) in progress_bar:
        x, labels = inputs.to(device), labels.to(device)
        x = torch.stack([x, # x.flip(-1), x.flip(-2), x.flip(-1, -2),
                        # x.transpose(-1,-2), x.transpose(-1,-2).flip(-1), x.transpose(-1,-2).flip(-2), x.transpose(-1,-2).flip(-1, -2),
                        # x.rot90(k=1, dims=[-1,-2]), x.rot90(k=2, dims=[-1,-2]), x.rot90(k=3, dims=[-1,-2])
                    ], 1).to(device)
        x = x.view(-1, 3, cfg['img_size'], cfg['img_size'])
        with torch.no_grad():
            outputs = net(x)
        outputs = outputs.view(inputs.shape[0], 1, -1).mean(1)
        preds.append(outputs.sigmoid().cpu().numpy())

        loss = criterion(outputs, labels)
        test_loss += loss.item()    

        description = 'Loss: %.3f' % (test_loss/(batch_idx+1))
        progress_bar.set_description(description)
    preds = np.concatenate(preds)
    auc, aucs = get_auc_score(val_labels, preds)

    test_loss /= len(val_loader)

    if auc > best_auc:
        best_auc = auc
        print(aucs)
        save_model(net, fold, auc, test_loss)

def main_loop(resume):
    os.makedirs('checkpoint', exist_ok=True)
    seed_everything(cfg['seed'])
    torch.cuda.empty_cache()

    folds = MultilabelStratifiedKFold(n_splits=cfg['fold_num'], shuffle=True, random_state=cfg['seed'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df[target_cols])

    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold < cfg['skip_fold']:
            continue
        global best_auc
        best_auc = 0.
        print(f'\n{fold}th fold training starts...')

        train_loader, _, val_loader, val_df = prepare_dataloader(train_idx, val_idx)

        net = CustomEffNet(pretrained=True)
        net = torch.nn.DataParallel(net, device_ids=range(cfg['device_num']))
        net = net.cuda()            

        model_arch = cfg['model_arch']

        if resume:
            checkpoint_path = f'checkpoint/siim-study-{model_arch}-fold{fold}-640.pth'
            checkpoint = torch.load(checkpoint_path)
            best_auc = checkpoint['auc']
            net.load_state_dict(checkpoint['net'])

        optimizer = optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'], last_epoch=-1)
        # criterion = nn.BCEWithLogitsLoss()
        criterion = LabelSmoothing()

        for epoch in range(cfg['epochs']):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(val_loader, net, criterion, fold, val_df[target_cols].values)

        del net, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=1)
args = parser.parse_args()

main_loop(args.resume)
