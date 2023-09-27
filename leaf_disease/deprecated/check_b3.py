import os
import cv2
import math
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

from efficientnet import *
import glob

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
    'fold_num': 10,
    'seed': 887,
    'model_arch': 'efficientnet-b4',
    'img_size': 384,
    'epochs': 3,
    'train_batch_size': 4,
    'val_batch_size': 8,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'device': 'cuda:0'
}

device = cfg['device']
best_acc = .0
best_loss = 0.1
label_df = pd.read_csv(label_path)
cudnn.benchmark = True

def get_img(img_path):
    img_bgr = cv2.imread(img_path)
    return img_bgr[..., ::-1]

def get_valid_transforms():
    return Compose([
            CenterCrop(cfg['img_size'], cfg['img_size'], p=1.),
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

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
        label = self.df['label'].values[idx]
        return img, label

def submission(test_loader, device, nets):
    result_dict = {}
    result_dict['image_id'] = []
    result_dict['label'] = []
    
    for i in range(len(nets)):
        nets[i].eval()
        nets[i].to(device)
    
    with torch.no_grad():
        cnt = 0
        for inputs, ids in test_loader:
            x = inputs.to(device)
            x = torch.stack([x,
                            transforms.RandomHorizontalFlip(p=1)(x),
                            transforms.RandomVerticalFlip(p=1)(x),
                            ], 0)
            x = x.view(-1, 3, cfg['img_size'], cfg['img_size'])
            preds = torch.zeros(inputs.shape[0], 5).to(device)
            for i in range(len(nets)):
                outputs = nets[i](x)
                outputs = outputs.view(inputs.shape[0], 3, -1).mean(1)
                preds += torch.softmax(outputs, dim=1)
            preds /= len(nets)
            preds = preds.argmax(dim=1)
            for idx, pred in enumerate(preds):
                result_dict['image_id'].append(ids[idx].item())
                result_dict['label'].append(pred.item())
    return pd.DataFrame(data=result_dict)

def check_models():
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    check_df = pd.read_csv(label_path)
    check_set = CassavaDataset(check_df, train_path, get_valid_transforms(), True)
    check_loader = torch.utils.data.DataLoader(check_set, batch_size=8, num_workers=8)

    nets = []
    for name in glob.glob('checkpoint/leaf-b3*.pth'):
        if 'fold9' in name:
            continue
        net = EfficientNet.from_name('efficientnet-b3').to(device)
        net._fc = nn.Linear(net._fc.in_features, 5)
        net = torch.nn.DataParallel(net)
        checkpoint = torch.load(name)
        net.load_state_dict(checkpoint['net'])
        nets.append(net)

    df = submission(check_loader, device, nets)
    print(sum(df.label.values == check_df.label.values) / check_df.shape[0])

check_models()