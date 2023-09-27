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
import matplotlib.pyplot as plt

project_path = os.path.abspath('.')
data_path = os.path.join(project_path, 'input')
train_path = os.path.join(data_path, 'train')
label_path = os.path.join(data_path, 'train.csv')
annotation_path = os.path.join(data_path, 'train_annotations.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')
teacher_path = os.path.join(project_path, 'teacher')

parser = argparse.ArgumentParser()

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_arch': 'resnet200d',
    'img_size': 448,
    'epochs': 20,
    'weights': [0.5, 1],
    'train_batch_size': 1,
    'val_batch_size': 1,
    'T_0': 5,
    'T_mul': 1,
    'lr': 5e-4,
    'min_lr': 1e-6,
    'accum_iter': 1,
    'weight_decay': 1e-6,
    'num_workers': 1,
    'device': 'cuda',
    'device_num': 1
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
            RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.9, 1.0)),
            HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.2),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),

            CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.2),
            Cutout(num_holes=5, max_h_size=int(0.1*cfg['img_size']), max_w_size=int(0.1*cfg['img_size']), p=0.2),

            # Normalize(mean=[0.4304, 0.4968, 0.3135], std=[0.2269, 0.2299, 0.2166], max_pixel_value=255.0, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ], additional_targets={'image_annot': 'image'}, p=1.0)

def get_check_transforms():
    return Compose([
            RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.9, 1.0)),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.2),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),

            CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.2),
            Cutout(num_holes=5, max_h_size=int(0.1*cfg['img_size']), max_w_size=int(0.1*cfg['img_size']), p=0.2),

            ToTensorV2(p=1.0)
        ], additional_targets={'image_annot': 'image'}, p=1.0)

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
    def __init__(self, df, df_annotation, anno_width=50, dPath=None, transforms=None, use_anno=False, test=False):
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
        image = get_img(img_path)
        label = torch.tensor(self.labels[idx]).float()

        if self.use_anno:
            image_annot = image.copy()
            query_string = f"StudyInstanceUID == '{uid}'"
            df = self.df_annotation.query(query_string)
            for _, row in df.iterrows():
                label = row['label']
                data = np.array(ast.literal_eval(row['data']))
                for d in data:
                    image_annot[d[1] - self.anno_width//2 : d[1] + self.anno_width//2,
                        d[0] - self.anno_width//2 : d[0] + self.anno_width//2,
                        :] = COLOR_MAP[label]

            augmented = self.transforms(image=image, image_annot=image_annot)
            img = augmented['image']
            img_anno = augmented['image_annot']
            return img, img_anno, label
        else:
            augmented = self.transforms(image=image)
            img = augmented['image']
        return img, label

class CustomResNet200D(nn.Module):
    def __init__(self, model_name=cfg['model_arch'], pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name)
        if pretrained:
            pretrained_path = './resnet200d_ra2-bdba9bf9.pth'
            self.model.load_state_dict(torch.load(pretrained_path))
            print('pretrained parameter loaded')
        
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11)

    def forward(self, x):
        features = self.model(x)
        pooled_features = self.pooling(features).view(x.size(0), -1)
        output = self.fc(pooled_features)
        return features, pooled_features, output

def check(train_idx, val_idx):
    check_df = label_df.loc[train_idx, :].reset_index(drop=True)
    check_df = check_df[check_df['StudyInstanceUID'].isin(df_annotation['StudyInstanceUID'].unique())].reset_index(drop=True)
    check_dataset = CatheterDataset(check_df, df_annotation=df_annotation, use_anno=True, dPath=train_path, transforms=get_check_transforms())
    for i in range(5):
        image, image_annot, label = check_dataset[i]
        plt.subplot(1, 2, 1)
        plt.imshow(image.transpose(0, 1).transpose(1, 2))
        plt.subplot(1, 2, 2)
        plt.imshow(image_annot.transpose(0, 1).transpose(1, 2))
        plt.title(f'label: {label}')
        plt.show()

def main_loop():
    folds = GroupKFold(n_splits=cfg['fold_num'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df[target_cols], label_df['PatientID'])

    for _, (train_idx, val_idx) in enumerate(folds):
        check(train_idx, val_idx)
        break

main_loop()
