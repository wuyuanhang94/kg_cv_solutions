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

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

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
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')

parser = argparse.ArgumentParser()

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_arch': 'resnet200d',
    'img_size': 448,
    'epochs': 20,
    'train_batch_size': 4,
    'val_batch_size': 8,
    'T_0': 5,
    'T_mul': 1,
    'lr': 5e-4,
    'min_lr': 1e-4,
    'accum_iter': 2,
    'weight_decay': 1e-6,
    'num_workers': 12,
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
            RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.9, 1.0)),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.1),
            RandomRotate90(p=0.1),

            OneOf([
                CLAHE(clip_limit=4.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ], p=0.1),

            OneOf([
                MotionBlur(blur_limit=5),
                MedianBlur(blur_limit=5),
                GaussianBlur(blur_limit=5),
                GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.1),

            OneOf([
                OpticalDistortion(distort_limit=0.8),
                GridDistortion(num_steps=5, distort_limit=0.8),
                ElasticTransform(alpha=3),
                IAAPiecewiseAffine(p=0.5),
                IAASharpen(p=0.5),
                IAAEmboss(p=0.5),
            ], p=0.1),

            OneOf([
                CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.5),
                Cutout(num_holes=5, max_h_size=int(0.1*cfg['img_size']), max_w_size=int(0.1*cfg['img_size']), p=0.5),
                GridDropout(p=0.4),
            ], p=0.1),

            Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ], p=1.0)

def get_valid_transforms():
    return Compose([
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

device = cfg['device']
cudnn.benchmark = True

best_auc = 0.
label_df = pd.read_csv(label_path)
target_cols = label_df.iloc[:, 1:-1].columns.tolist()

class CatheterDataset(Dataset):
    def __init__(self, df, dPath=None, transforms=None, test=False):
        super(CatheterDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.dPath = dPath
        self.transforms = transforms
        self.test = test
        self.labels = df[target_cols].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = self.df['StudyInstanceUID'].values[idx] + '.jpg'
        img_path = os.path.join(self.dPath, img_name)
        img = get_img(img_path)
        img = self.transforms(image=img)['image']
        if self.test:
            label = img_name
        else:
            label = torch.tensor(self.labels[idx]).float()
        return img, label

def prepare_dataloader(label_df, train_idx, val_idx):
    train_df = label_df.loc[train_idx, :].reset_index(drop=True)
    val_df = label_df.loc[val_idx, :].reset_index(drop=True)

    train_dataset = CatheterDataset(train_df, train_path, get_train_transforms())
    val_dataset = CatheterDataset(val_df, train_path, get_valid_transforms())
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    return train_loader, val_loader

def save_model(net, fold, auc=.0, loss=.0):
    state = {
        'net': net.state_dict(),
        'auc': auc,
        'loss': loss
    }
    print(f'Saving auc: {auc}...')
    model_arch = cfg['model_arch']
    torch.save(state, os.path.join(project_path, f'checkpoint/catheter-{model_arch}-fold{fold}-auc.pth'))

def get_auc_score(labels, outputs):
    auc = []
    for j in range(labels.shape[1]):
        auc.append(roc_auc_score(labels[:, j], outputs[:, j]))
    avg_auc = np.mean(auc)
    return avg_auc

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0.
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        outputs = torch.sigmoid(outputs)
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

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            x, labels = inputs.to(device), labels.to(device)
            x = torch.stack([x, # x.flip(-1), x.flip(-2), x.flip(-1, -2),
                            # x.transpose(-1,-2), x.transpose(-1,-2).flip(-1), x.transpose(-1,-2).flip(-2), x.transpose(-1,-2).flip(-1, -2),
                            # x.rot90(k=1, dims=[-1,-2]), x.rot90(k=2, dims=[-1,-2]), x.rot90(k=3, dims=[-1,-2])
                        ], 1).to(device)
            x = x.view(-1, 3, cfg['img_size'], cfg['img_size'])
            outputs = net(x)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(inputs.shape[0], 1, -1).mean(1)
            preds.append(outputs.detach().cpu().numpy())

            loss = criterion(outputs, labels)
            test_loss += loss.item()    

            description = 'Loss: %.3f' % (test_loss/(batch_idx+1))
            progress_bar.set_description(description)
    preds = np.concatenate(preds)
    auc = get_auc_score(val_labels, preds)

    test_loss /= len(val_loader)

    if auc > best_auc:
        best_auc = auc
        save_model(net, fold, auc, test_loss)

def main_loop(resume, adam):
    os.makedirs('checkpoint', exist_ok=True)
    seed_everything(cfg['seed'])
    torch.cuda.empty_cache()

    folds = GroupKFold(n_splits=cfg['fold_num'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df[target_cols], label_df['PatientID'])

    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold < 0:
            continue
        global best_auc
        best_auc = 0.
        print(f'\n{fold}th fold training starts...')

        train_loader, val_loader = prepare_dataloader(label_df, train_idx, val_idx)

        if resume:
            net = timm.create_model(cfg['model_arch'], pretrained=False, num_classes=11).to(device)
            net.fc = nn.Linear(net.fc.in_features, 11)
            net = torch.nn.DataParallel(net, device_ids=[0])
            net = net.cuda()            

            model_arch = cfg['model_arch']
            checkpoint = torch.load(f'checkpoint/catheter-{model_arch}-fold{fold}-auc.pth')
            best_auc = checkpoint['auc']

            pretrained_dict = checkpoint['net']
            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
        else:
            if not os.path.exists('resnet200d_ra2-bdba9bf9.pth'):
                import wget
                wget.download('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth')
            net = timm.create_model(cfg['model_arch'], pretrained=False)
            checkpoint = torch.load('resnet200d_ra2-bdba9bf9.pth')
            net.load_state_dict(checkpoint)
            net.fc = nn.Linear(net.fc.in_features, 11)

            net = torch.nn.DataParallel(net, device_ids=[0])
            net = net.cuda()

        if adam:
            optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'], last_epoch=-1)
        criterion = nn.BCELoss()

        for epoch in range(cfg['epochs']):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)

            val_df = label_df.loc[val_idx, :].reset_index(drop=True)
            val_labels = val_df[target_cols].values
            validate(val_loader, net, criterion, fold, val_labels)

        del net, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=0)
parser.add_argument('--Adam', type=int, help='use Adam or SGD', default=1)
args = parser.parse_args()

main_loop(args.resume, args.Adam)
