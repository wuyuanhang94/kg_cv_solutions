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
best_loss = 0.39
label_df = pd.read_csv(label_path)
cudnn.benchmark = True

import os
import cv2
import tqdm
import random
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

data_path = os.path.abspath(os.path.join(os.path.pardir, 'input/cassava-leaf-disease-classification'))
train_path = os.path.join(data_path, 'train_images')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test_images')
csv_path = os.path.join(data_path, 'submission.csv')
cwd = os.path.abspath('.')
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
        # img = np.array(img)
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

torch.cuda.empty_cache()
cfg = {
    'fold_num': 5,
    'seed': 899,
    'model_arch': 'efficientnet-b6',
    'img_size': 384,
    'epochs': 8,
    'train_batch_size': 4,
    'val_batch_size': 8,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 1,
    'device': 'cuda:0'
}

def submission(csv_path, test_loader, device, nets):
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
#             x = torch.stack([x,
#                             transforms.RandomHorizontalFlip(p=1)(x),
#                             transforms.RandomVerticalFlip(p=1)(x),
#                             ], 0)
#             x = x.view(-1, 3, cfg['img_size'], cfg['img_size'])
            # print(x.shape)
            preds = torch.zeros(inputs.shape[0], 5).to(device)
            for i in range(len(nets)):
                outputs = nets[i](x)
                #outputs = outputs.view(inputs.shape[0], 2, -1).mean(1)
                preds += torch.softmax(outputs, dim=1)
                # print(preds/(i+1))
            preds /= len(nets)
            # print(preds)
            preds = preds.argmax(dim=1)
            for idx, pred in enumerate(preds):
                result_dict['image_id'].append(ids[idx].item())
                result_dict['label'].append(pred.item())
            cnt += 1
            if cnt >= 25:
                break
    result_df = pd.DataFrame(data=result_dict)
    submission_path = 'submission.csv'
    result_df.to_csv(submission_path, index=None)
    return result_df

def check_models():
    device = torch.device('cuda')
    check_df = pd.read_csv(label_path)
    check_set = CassavaDataset(check_df, train_path, get_valid_transforms(), True)
    check_loader = torch.utils.data.DataLoader(check_set, batch_size=8, num_workers=8)

    nets = []
    nets_name = os.listdir('checkpoint/')
    for idx, name in enumerate(nets_name):
        if idx < 10:
            print(name)
            net = EfficientNet.from_name('efficientnet-b4').to(device)
            net._fc = nn.Linear(net._fc.in_features, 5)
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load(os.path.join('checkpoint', name))
            net.load_state_dict(checkpoint['net'])
            nets.append(net)

    df = submission(csv_path, check_loader, device, nets)

check_models()