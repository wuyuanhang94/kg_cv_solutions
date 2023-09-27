import os
import cv2
import ast
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

import matplotlib.pyplot as plt

project_path = os.path.abspath('.')
data_path = os.path.join(project_path, 'input')
train_path = os.path.join(data_path, 'train')
label_path = os.path.join(data_path, 'train.csv')
annotation_path = os.path.join(data_path, 'train_annotations.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')

def get_img(img_path):
    img_bgr = cv2.imread(img_path)
    return img_bgr[..., ::-1]

def get_train_transforms():
    return Compose([
            # Resize(512, 512),
            RandomResizedCrop(512, 512, scale=(0.9, 1.0)),
            HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
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
                CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.2),
                Cutout(num_holes=5, max_h_size=int(0.1*512), max_w_size=int(0.1*512), p=0.2),
            ], p=1.0),

            # Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ToTensorV2(p=1.0)
        ], p=1.0)

def get_valid_transforms():
    return Compose([
            Resize(512, 512),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            # ToTensorV2(p=1.0),
        ], p=1.0)

device = torch.device('cuda')
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
        print(uid)
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

def prepare_dataloader(train_idx, val_idx):
    train_df = label_df.loc[train_idx, :].reset_index(drop=True)
    train_df = train_df[train_df['StudyInstanceUID'].isin(df_annotation['StudyInstanceUID'].unique())].reset_index(drop=True)

    val_df = label_df.loc[val_idx, :].reset_index(drop=True)
    val_df = val_df[val_df['StudyInstanceUID'].isin(df_annotation['StudyInstanceUID'].unique())].reset_index(drop=True)

    train_dataset = CatheterDataset(df=train_df, df_annotation=df_annotation, dPath=train_path, transforms=get_train_transforms())
    val_dataset = CatheterDataset(df=val_df, df_annotation=df_annotation, dPath=train_path, transforms=get_valid_transforms())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader

def main_loop():
    folds = GroupKFold()
    folds = folds.split(np.arange(label_df.shape[0]), label_df[target_cols], label_df['PatientID'])

    for train_idx, val_idx in folds:
        train_loader, val_loader = prepare_dataloader(train_idx, val_idx)
        for images, labels in train_loader:
            plt.imshow(images.squeeze())
            plt.show()
            break

main_loop()
