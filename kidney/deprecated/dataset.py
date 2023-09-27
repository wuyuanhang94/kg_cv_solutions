import os
import sys
import cv2
import argparse
from tqdm import tqdm
import warnings
# warnings.filterwarnings("ignore")

import numpy as np
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

project_path = os.path.abspath('.')
data_path = os.path.join(project_path, 'train_tiles')
images_path = os.path.join(data_path, 'images')
masks_path = os.path.join(data_path, 'masks')

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model': 'resnet34',
    'img_size': 384,
    'epochs': 20,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'T_0': 1,
    'T_mul': 2,
    'lr': 2e-4,
    'min_lr': 2e-6,
    'accum_iter': 2,
    'weight_decay': 1e-6,
    'num_workers': 8,
    'device': 'cuda'
}
best_loss = float('inf')

train_transforms = A.Compose([
    A.Resize(cfg['img_size'], cfg['img_size']),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.HueSaturationValue(p=0.5),
    A.Rotate(p=0.5),
    A.RandomBrightnessContrast(p=0.5),

    A.OneOf([
        A.OpticalDistortion(distort_limit=0.8),
        A.GridDistortion(num_steps=3, distort_limit=0.5),
        A.ElasticTransform(alpha=3),
        A.IAAPiecewiseAffine(p=0.5),
        A.IAASharpen(p=0.5),
        A.IAAEmboss(p=0.5),
    ], p=0.5),

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

def test_aug():
    dataset = KidneyDataset(images_path=images_path, masks_path=masks_path, transform=train_transforms)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    cnt = 6
    samples = np.random.randint(0, len(dataset), cnt)
    _, ax = plt.subplots(nrows=2, ncols=cnt, figsize=(18, 10))
    for i, idx in enumerate(samples):
        img, mask, _ = dataset[idx]
        ax[0, i].imshow(img)
        ax[1, i].imshow(mask)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_aug()
