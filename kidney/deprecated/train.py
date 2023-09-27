import os
import sys
import cv2
import argparse
from tqdm import tqdm

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from sklearn.model_selection import GroupKFold

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
best_loss = float('inf')
best_dice = 0.767

train_transforms = A.Compose([
    A.Resize(cfg['img_size'], cfg['img_size']),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.05, rotate_limit=30, p=0.5),
    # A.Rotate(p=0.5),

    A.OneOf([
        A.CLAHE(clip_limit=4.0, p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], p=0.3),

    A.OneOf([
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        A.GridDistortion(num_steps=3, distort_limit=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.IAAPiecewiseAffine(p=0.5),
        A.IAASharpen(p=0.5),
        A.IAAEmboss(p=0.5),
    ], p=0.3),

    A.Normalize(mean=(0.5936, 0.4990, 0.6150), std=(0.0757, 0.0967, 0.0623)),
    ToTensorV2(),
], p=1.0)

val_transforms = A.Compose([
    A.Resize(cfg['img_size'], cfg['img_size']),
    A.Normalize(mean=(0.5936, 0.4990, 0.6150), std=(0.0757, 0.0967, 0.0623)),
    ToTensorV2(),
], p=1.0)

class KidneyDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.images = os.listdir(images_path)
        self.masks_path = masks_path
        self.masks = os.listdir(masks_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_id = self.images[idx].split('.')[0]
        mask_path = os.path.join(self.masks_path, img_id+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        seg = np.array(mask, dtype=np.uint8).max() == 1

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return (img, mask, 1) if seg else (img, mask, 0)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # brings some stochasticity, but 10% speed up

full_dataset = KidneyDataset(images_path=images_path, masks_path=masks_path, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
validate_size = len(full_dataset) - train_size

train_set, validate_set = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=8)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=cfg['val_batch_size'], num_workers=8)

aux_params=dict(
    pooling='avg',            
    dropout=0.5,              
    activation='sigmoid',     
    classes=1,                
)
net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid', aux_params=aux_params)
# net = torch.nn.DataParallel(net, device_ids=range(2) if device_name == 'TITAN V' else range(1))
checkpoint = torch.load('checkpoint/kidney-unet.pth')
net.load_state_dict(checkpoint['net'])
net = net.to('cuda')

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

def prepare_dataloader(train_idx, val_idx):


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

def main_loop(resume, adam):
    seed_everything(cfg['seed'])
    start_epoch = 0
    torch.cuda.empty_cache()

    folds = GroupKFold(n_splits=cfg['fold_num'])
    folds = folds.split(np.arange(label_df.shape[0]), label_df)
    
    for epoch in range(start_epoch, start_epoch+50):
        print('\nEpoch: %d' % epoch)
        train(epoch)
        validate(epoch)

parser.add_argument('--resume', type=int, help='starting from checkpoint or pretrained', default=1)
parser.add_argument('--Adam', type=int, help='use Adam or SGD', default=1)
args = parser.parse_args()

main_loop(args.resume, args.Adam)
