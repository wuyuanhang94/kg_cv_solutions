import os
import cv2
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from sklearn.model_selection import KFold

cwd = os.path.abspath('.')
data_path = os.path.join(cwd, 'hubmap512')
images_path = os.path.join(data_path, 'train')
masks_path = os.path.join(data_path, 'masks')
label_path = os.path.join(data_path, 'train.csv')

parser = argparse.ArgumentParser()
device_name = torch.cuda.get_device_name()

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_name': 'fpn',
    'encoder_name': 'timm-efficientnet-b4',
    'aug_classifier': False,
    'img_size': 512,
    'epochs': 6,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'T_0': 3,
    'T_mul': 1,
    'lr': 1e-4,
    'min_lr': 1e-5,
    'loss_weights': [0.8, 0.2], # dice, bce
    'accum_iter': 1,
    'weight_decay': 1e-6,
    'num_workers': 12,
    'device': 'cuda',
    'skip_fold': 0,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True brings some stochasticity, but 10% speed up

best_dice = .0
mean = np.array([0.63759809, 0.4716141, 0.68231112])
std = np.array([0.16475244, 0.22850685, 0.14593643])

def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT),
    A.Transpose(p=0.5),

    A.OneOf([
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], p=0.5),

    A.OneOf([
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
        A.GridDistortion(num_steps=3, distort_limit=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    ], p=0.5),

    A.OneOf([
        A.IAAPiecewiseAffine(p=0.5),
        A.IAASharpen(p=0.5),
        A.IAAEmboss(p=0.5),
    ], p=0.5),
], p=1.0)

class HuBMAPDataset(Dataset):
    def __init__(self, ids, train=True, tfms=None):
        self.fnames = [fname for fname in os.listdir(images_path) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(images_path, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(masks_path, fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        return img2tensor((img / 255.0 - mean) / std), img2tensor(mask)

def prepare_dataloader(train_ids, val_ids):
    train_dataset = HuBMAPDataset(train_ids, train=True, tfms=train_transforms)
    val_dataset = HuBMAPDataset(val_ids, train=False, tfms=None)
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    return train_loader, val_loader

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return dice_loss, bce_loss

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_prop = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - bce_prop) ** self.gamma * bce_loss

        return dice_loss, focal_loss

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0.
    train_dice = 0.

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, masks) in progress_bar:
        inputs, masks = inputs.to(cfg['device']), masks.to(cfg['device'], dtype=torch.float32)
        
        optimizer.zero_grad()
        y_preds = net(inputs)
        dice_loss, bce_loss = criterion(y_preds, masks)
        loss = cfg['loss_weights'][0] * dice_loss + cfg['loss_weights'][1] * bce_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        dice_ceff = 1 - dice_loss.item()
        train_dice += dice_ceff
        description = 'Loss: %.4f | Dice: %.4f' % (train_loss / (batch_idx+1), train_dice / (batch_idx+1))
        progress_bar.set_description(description)
    
    scheduler.step()

def validate(fold, val_loader, net, criterion):
    net.eval()
    val_loss = 0.
    val_dice = 0.
    global best_dice

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for batch_idx, (inputs, masks) in progress_bar:
            inputs, masks = inputs.to(cfg['device']), masks.to(cfg['device'], dtype=torch.float32)
            
            y_preds = net(inputs)
            dice_loss, bce_loss = criterion(y_preds, masks)
            loss = cfg['loss_weights'][0] * dice_loss + cfg['loss_weights'][1] * bce_loss
            val_loss += loss.item()
            dice_ceff = 1 - dice_loss.item()
            val_dice += dice_ceff
            description = 'Loss: %.4f | Dice: %.4f' % (val_loss / (batch_idx+1), val_dice / (batch_idx+1))
            progress_bar.set_description(description)
            
        if val_dice / (batch_idx + 1) > best_dice:
            best_dice = val_dice / (batch_idx + 1)
            val_loss = val_loss / (batch_idx + 1)
            print(f'best dice: {best_dice}, loss: {val_loss}, saving model...')
            state = {
                'net': net.state_dict(),
                'loss': val_loss,
                'dice': best_dice,
            }
            model_name = cfg['model_name']
            encoder_name = cfg['encoder_name']
            img_size = cfg['img_size']
            torch.save(state, f'checkpoint/hubmap-{model_name}-{encoder_name}-{img_size}-fold{fold}-dice.pth')

def main_loop(resume, adam):
    seed_everything(cfg['seed'])
    torch.cuda.empty_cache()
    os.makedirs('checkpoint', exist_ok=True)

    ids = pd.read_csv(label_path).id.values
    folds = KFold(n_splits=cfg['fold_num'], random_state=cfg['seed'], shuffle=True).split(ids)
    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold < cfg['skip_fold']:
            continue
        global best_dice
        best_dice = .0
        print(f'\n{fold}th fold training starts...')

        train_ids, val_ids = ids[train_idx], ids[val_idx]
        train_loader, val_loader = prepare_dataloader(train_ids, val_ids)

        aux_params = None
        if cfg['aug_classifier']:
            aux_params=dict(
                pooling='avg',            
                dropout=0.5,              
                activation='sigmoid',  
                classes=1,               
            )
        net = smp.FPN(encoder_name=cfg['encoder_name'], encoder_weights='imagenet', classes=1, activation=None, aux_params=aux_params)
        if resume:
            model_name = cfg['model_name']
            encoder_name = cfg['encoder_name']
            img_size = cfg['img_size']
            checkpoint = torch.load(f'checkpoint/hubmap-{model_name}-{encoder_name}-{img_size}-fold{fold}-dice.pth')
            best_dice = checkpoint['dice']
            net.load_state_dict(checkpoint['net'])

        net = net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'])
        # criterion = DiceBCELoss()
        criterion = DiceFocalLoss()

        for epoch in range(cfg['epochs']):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(fold, val_loader, net, criterion)

        del net, train_loader, val_loader, optimizer, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=0)
parser.add_argument('--Adam', type=int, help='use Adam or SGD', default=1)
args = parser.parse_args()

main_loop(args.resume, args.Adam)
