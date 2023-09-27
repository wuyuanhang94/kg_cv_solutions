# import sys
# sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
# import timm
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

cwd = os.path.abspath('.')
data_path = os.path.abspath(os.path.join(cwd, 'input'))
train_path = os.path.join(data_path, 'train')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')

cudnn.benchmark = True

cfg = {
    'model_arch': 'resnet200d',
    'img_size': 512,
    'batch_size': 32,
    'target_size': 11,
    'target_cols': ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                    'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                    'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                    'Swan Ganz Catheter Present'],
    'device': 'cuda',
}
device = cfg['device']

def get_img(img_path):
    img_bgr = cv2.imread(img_path)
    return img_bgr[..., ::-1]

def get_valid_transforms():
    return Compose([
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ], p=1.0)

class CatheterDataset(Dataset):
    def __init__(self, df, dPath=None, transforms=None):
        super(CatheterDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.dPath = dPath
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        uid = self.df['StudyInstanceUID'].values[idx]
        img_name = uid + '.jpg'
        img_path = os.path.join(self.dPath, img_name)
        image = get_img(img_path)
        augmented = self.transforms(image=image)
        img = augmented['image']
        return img

class CustomResNet200D(nn.Module):
    def __init__(self, model_name=cfg['model_arch']):
        super().__init__()
        self.model = timm.create_model(model_name)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11)

    def forward(self, x):
        features = self.model(x)
        pooled_features = self.pooling(features).view(x.size(0), -1)
        output = self.fc(pooled_features)
        return output

def inference(test_loader, nets):
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for net in nets:
        net.eval()
        net.to(device)

    for batch_idx, images in progress_bar:
        images = images.to(device)
        avg_preds = []
        for net in nets:
            with torch.no_grad():
                y_preds1 = net(images)
                y_preds2 = net(images.flip(-1))
            y_preds = (y_preds1.sigmoid().to('cpu').numpy() + y_preds2.sigmoid().to('cpu').numpy()) / 2
            avg_preds.append(y_preds)
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

def submission():
    torch.cuda.empty_cache()

    test_df = pd.read_csv(csv_path, nrows=10)
    test_set = CatheterDataset(test_df, test_path, get_valid_transforms())
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=12)

    nets = []
    dire = '/home/yi/stage3'
    for name in os.listdir(dire):
        net = CustomResNet200D()
        net = torch.nn.DataParallel(net)
        checkpoint = torch.load(os.path.join(dire, name))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        nets.append(net)
    
    predictions = inference(test_loader, nets)
    test_df[cfg['target_cols']] = predictions
    test_df[['StudyInstanceUID'] + cfg['target_cols']].to_csv('submission.csv', index=False)

submission()