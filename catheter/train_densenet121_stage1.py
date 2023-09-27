import os
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
annotation_path = os.path.join(data_path, 'train_annotations.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')

parser = argparse.ArgumentParser()

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_arch': 'densenet121',
    'img_size': 640,
    'epochs': 2,
    'train_batch_size': 6,
    'val_batch_size': 8,
    'T_0': 2,
    'T_mul': 1,
    'lr': 1e-4,
    'min_lr': 1e-5,
    'accum_iter': 1,
    'weight_decay': 1e-6,
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
            RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.9, 1.0)),
            HorizontalFlip(p=0.5),
            
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.2),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),

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

            CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.2),
            Cutout(num_holes=5, max_h_size=int(0.1*cfg['img_size']), max_w_size=int(0.1*cfg['img_size']), p=0.2),

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
cudnn.benchmark = True

best_auc = 0.
label_df = pd.read_csv(label_path)
df_annotation = pd.read_csv(annotation_path)
target_cols = label_df.iloc[:, 1:-1].columns.tolist()

COLOR_MAP = {
    'ETT - Abnormal': (255, 0, 0),
    'ETT - Borderline': (0, 255, 0),
    'ETT - Normal': (0, 0, 255),
    'NGT - Abnormal': (255, 255, 0),
    'NGT - Borderline': (255, 0, 255),
    'NGT - Incompletely Imaged': (0, 255, 255),
    'NGT - Normal': (128, 0, 0),
    'CVC - Abnormal': (0, 128, 0),
    'CVC - Borderline': (0, 0, 128),
    'CVC - Normal': (128, 128, 0),
    'Swan Ganz Catheter Present': (128, 0, 128),
}

class CatheterDataset(Dataset):
    def __init__(self, df, df_annotation, anno_width=50, dPath=None, transforms=None, use_anno=True, test=False):
        super(CatheterDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.df_annotation = df_annotation
        self.anno_width = anno_width
        self.dPath = dPath
        self.transforms = transforms
        self.use_anno = use_anno
        self.test = test
        self.labels = df[target_cols].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        uid = self.df['StudyInstanceUID'].values[idx]
        img_name = uid + '.jpg'
        img_path = os.path.join(self.dPath, img_name)
        img = get_img(img_path)

        query_string = f"StudyInstanceUID == '{uid}'"
        df = self.df_annotation.query(query_string)
        for i, row in df.iterrows():
            label = row['label']
            data = np.array(ast.literal_eval(row['data']))
            for d in data:
                img[d[1] - self.anno_width//2 : d[1] + self.anno_width//2,
                    d[0] - self.anno_width//2 : d[0] + self.anno_width//2,
                    :] = COLOR_MAP[label]

        img = self.transforms(image=img)['image']
        if self.test:
            label = img_name
        else:
            label = torch.tensor(self.labels[idx]).float()
        return img, label

class CustomDensenet121(nn.Module):
    def __init__(self, model_name=cfg['model_arch']):
        super().__init__()
        self.model = timm.create_model(model_name)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11)

    def forward(self, x):
        features = self.model(x)
        pooled_features = self.pooling(features).view(x.size(0), -1)
        output = self.fc(pooled_features)
        return features, pooled_features, output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, labels):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, labels, reduce=False)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * bce_loss
        return F_loss.mean()

def prepare_dataloader(train_idx, val_idx):
    train_df = label_df.loc[train_idx, :].reset_index(drop=True)
    train_df = train_df[train_df['StudyInstanceUID'].isin(df_annotation['StudyInstanceUID'].unique())].reset_index(drop=True)
    val_df = label_df.loc[val_idx, :].reset_index(drop=True)
    val_df = val_df[val_df['StudyInstanceUID'].isin(df_annotation['StudyInstanceUID'].unique())].reset_index(drop=True)

    train_dataset = CatheterDataset(df=train_df, df_annotation=df_annotation, dPath=train_path, transforms=get_train_transforms())
    val_dataset = CatheterDataset(df=val_df, df_annotation=df_annotation, dPath=train_path, transforms=get_valid_transforms())
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
    torch.save(state, os.path.join(project_path, f'teacher/catheter-{model_arch}-teacher-fold{fold}-auc.pth'))

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
    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        _, _, outputs = net(inputs)
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
            _, _, outputs = net(x)
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

def main_loop(resume, adam):
    os.makedirs('teacher', exist_ok=True)
    seed_everything(cfg['seed'])
    torch.cuda.empty_cache()

    folds = GroupKFold(n_splits=cfg['fold_num'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df[target_cols], label_df['PatientID'])

    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold < 2:
            continue
        global best_auc
        best_auc = 0.
        print(f'\n{fold}th fold training starts...')

        train_loader, _, val_loader, val_df = prepare_dataloader(train_idx, val_idx)

        if resume:
            net = CustomDensenet121()
            net = torch.nn.DataParallel(net, device_ids=[0])
            net = net.cuda()            

            model_arch = cfg['model_arch']
            checkpoint = torch.load(f'teacher/catheter-{model_arch}-teacher-fold{fold}-auc.pth')
            best_auc = checkpoint['auc']

            pretrained_dict = checkpoint['net']
            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
        else:
            net = CustomDensenet121()
            pretrained_path = './densenet121_chestx.pth'
            pretrained_checkpoint = torch.load(pretrained_path)
            
            pretrained_dict = pretrained_checkpoint['model']
            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
            net = net.cuda()
            print('pretrained parameter loaded')

        if adam:
            optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'], last_epoch=-1)
        criterion = FocalLoss()

        for epoch in range(cfg['epochs']):
            print('\nEpoch: %d' % epoch)
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(val_loader, net, criterion, fold, val_df[target_cols].values)

        del net, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=0)
parser.add_argument('--Adam', type=int, help='use Adam or SGD', default=1)
args = parser.parse_args()

main_loop(args.resume, args.Adam)
