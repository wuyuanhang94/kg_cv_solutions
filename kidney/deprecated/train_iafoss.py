import os
import sys
import cv2
import argparse
from tqdm import tqdm

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from sklearn.model_selection import GroupKFold, KFold

cwd = os.path.abspath('.')
data_path = os.path.join(cwd, 'train_tiles')
images_path = os.path.join(data_path, 'images')
masks_path = os.path.join(data_path, 'masks')

parser = argparse.ArgumentParser()
device_name = torch.cuda.get_device_name()

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model': 'resnet34',
    'img_size': 512,
    'epochs': 20,
    'train_batch_size': 4,
    'val_batch_size': 4,
    'T_0': 1,
    'T_mul': 2,
    'lr': 1e-3,
    'min_lr': 1e-4,
    'accum_iter': 1,
    'weight_decay': 1e-6,
    'num_workers': 8,
    'device': 'cuda'
}

bs = 64
nfolds = 4
fold = 0
SEED = 2020
TRAIN = 'hubmap256/train/'
MASKS = 'hubmap256/masks/'
LABELS = 'input/train.csv'
NUM_WORKERS = 4

best_loss = float('inf')
best_dice = 0.767

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # brings some stochasticity, but 10% speed up

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim == 2: img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self, fold=fold, train=True, tfms=None):
        ids = pd.read_csv(LABELS).id.values
        kf = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])#难理解
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS, fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img, mask

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT),
    # A.Transpose(p=0.5),
    # A.Rotate(p=0.5),

    A.OneOf([
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], p=0.5),

    A.OneOf([
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        A.GridDistortion(num_steps=3, distort_limit=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.IAAPiecewiseAffine(p=0.5),
        A.IAASharpen(p=0.5),
        A.IAAEmboss(p=0.5),
    ], p=0.3),

    # A.Normalize(mean=(0.65459856, 0.48386562, 0.69428385), std=(0.15167958, 0.23584107, 0.13146145)),
    ToTensorV2(),
], p=1.0)

val_transforms = A.Compose([
    A.Normalize(mean=(0.65459856, 0.48386562, 0.69428385), std=(0.15167958, 0.23584107, 0.13146145)),
    ToTensorV2(),
], p=1.0)

aux_params=dict(
    pooling='avg',            
    dropout=0.5,              
    activation='sigmoid',     
    classes=1,                
)
net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid', aux_params=aux_params)
# net = torch.nn.DataParallel(net, device_ids=range(2) if device_name == 'TITAN V' else range(1))
# checkpoint = torch.load('checkpoint/kidney-unet.pth')
# net.load_state_dict(checkpoint['net'])
# net = net.to('cuda')

optimizer = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'])

def dice_ceff(y_pred, y_gt):
    smooth = 0.01
    probs = y_pred
    num = y_gt.size(0)
    predicts = (probs.view(num, -1) > 0.35).float()
    y_gt = y_gt.view(num, -1)
    intersection = predicts * y_gt
    score = (2.0 * intersection.sum(1) + smooth) / (predicts.sum(1) + y_gt.sum(1) + smooth)
    return score.mean()

def dice_loss(y_pred, y_gt):
    return 1 - dice_ceff(y_pred, y_gt)

def bce_dice_loss(y_pred, y_gt):
    bceL = nn.BCELoss()(y_pred.view(-1), y_gt.view(-1))
    diceL = dice_loss(y_pred, y_gt)
    return 0.75*bceL + 0.25*diceL

def train(epoch):
    net.train()
    train_loss = 0.
    train_dice = 0.
    train_seg = 0.
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, masks, labels) in progress_bar:
        inputs, masks, labels = inputs.to(cfg['device'], dtype=torch.float32), masks.to(cfg['device'], dtype=torch.float32), labels.to(cfg['device'], dtype=torch.float32)
        
        optimizer.zero_grad()
        y_pred, logits = net(inputs)
        loss = nn.BCELoss()(logits.squeeze(), labels)
        
        if max(labels) == 1:
            seg_idx = labels == 1
            train_seg += sum(seg_idx)
            loss += bce_dice_loss(y_pred[seg_idx, ...], masks[seg_idx, ...])
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if max(labels) == 1:
            dice = dice_ceff(y_pred[seg_idx, ...], masks[seg_idx, ...])
            train_dice += dice*sum(seg_idx)
            description = 'Loss: %.3f | Dice: %.3f' % (train_loss / (batch_idx+1), train_dice / train_seg)
            progress_bar.set_description(description)

def validate(epoch):
    net.eval()
    test_loss = 0.
    test_dice = 0.
    test_seg = 0.
    global best_loss
    global best_dice

    with torch.no_grad():
        progress_bar = tqdm(enumerate(validate_loader), total=len(validate_loader))
        for batch_idx, (inputs, masks, labels) in progress_bar:
            inputs, masks, labels = inputs.to(cfg['device'], dtype=torch.float32), masks.to(cfg['device'], dtype=torch.float32), labels.to(cfg['device'], dtype=torch.float32)
            
            y_pred, logits = net(inputs)
            loss = nn.BCELoss()(logits.squeeze(), labels)

            if max(labels) == 1:
                seg_idx = labels == 1
                test_seg += sum(seg_idx)
                loss += bce_dice_loss(y_pred[seg_idx, ...], masks[seg_idx, ...])

            test_loss += loss.item()
            if max(labels) == 1:
                dice = dice_ceff(y_pred[seg_idx, ...], masks[seg_idx, ...])
                test_dice += dice*sum(seg_idx)
                description = 'Loss: %.3f | Dice: %.3f' % (test_loss / (batch_idx+1), test_dice / test_seg)
                progress_bar.set_description(description)
            
        if test_dice/test_seg > best_dice:
            best_loss = test_loss / (batch_idx+1)
            os.makedirs('checkpoint', exist_ok=True)
            print('Saving model...')
            state = {
                'net': net.state_dict(),
                'loss': best_loss,
                'dice': test_dice/test_seg,
                'epoch': epoch
            }
            torch.save(state, 'checkpoint/kidney-unet.pth')

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6,12,18,24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU()
        )
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(
            m.conv1,
            m.bn1,
            nn.ReLU(inplace=True)
        )
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
            m.layer1
        )
        self.enc2 = m.layer2
        self.enc3 = m.layer3
        self.enc4 = m.layer4

        self.aspp = 


def dataset_plt():
    seed_everything(cfg['seed'])
    ds = HuBMAPDataset(tfms=train_transforms)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    imgs, masks = next(iter(dl))

    plt.figure(figsize=(10, 16))
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        img = img.permute(1,2,0).numpy().astype(np.uint8)
        plt.subplot(8, 8, i+1)
        plt.imshow(img, vmin=0, vmax=255)
        plt.imshow(mask.squeeze().numpy(), alpha=0.2)
        plt.axis('off')
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.show()
    del ds, dl, imgs, masks

def main_loop(resume, adam):
    seed_everything(cfg['seed'])
    start_epoch = 0
    torch.cuda.empty_cache()

    ds = HuBMAPDataset(tfms=train_transforms)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    
    for epoch in range(start_epoch, start_epoch+50):
        print('\nEpoch: %d' % epoch)
        train(epoch)
        validate(epoch)

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=1)
parser.add_argument('--Adam', type=int, help='use Adam or SGD', default=1)
args = parser.parse_args()

# main_loop(args.resume, args.Adam)
dataset_plt()
