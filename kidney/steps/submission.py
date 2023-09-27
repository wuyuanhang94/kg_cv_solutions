import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
from tqdm.notebook import tqdm
import zipfile
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc
import glob
import segmentation_models_pytorch as smp

from albumentations import (Compose, Normalize)
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_grid(shape, window=1024, min_overlap=256):
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

sz = 512
reduce = 2
min_overlap = 256
s_th = 40
p_th = 1000 * (sz//256) ** 2
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

DATA = 'input/test/'
df_sample = pd.read_csv('input/sample_submission.csv')
device = 'cuda'

mean_512 = np.array([0.63990417, 0.4734721, 0.68480998])
std_512 = np.array([0.16061672, 0.22722983, 0.14034663])

def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HubMAPDataset(Dataset):
    def __init__(self, idx, sz, reduce, mean, std):
        self.data = rasterio.open(os.path.join(DATA, idx+'.tiff'), transform=identity, num_threads='all_cpus')
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce * sz
        self.mean = mean
        self.std = std
        self.make_grid = make_grid(self.shape, window=self.sz, min_overlap=min_overlap)
        
    def __len__(self):
        return len(self.make_grid)
    
    def __getitem__(self, idx):
        img = np.zeros((self.sz, self.sz, 3), np.uint8)

        x1, x2, y1, y2 = self.make_grid[idx]
        if self.data.count == 3:
            img = self.data.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
            img = np.moveaxis(img, 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[:, :, i] = layer.read(1, window=Window.from_slices((x1, x2), (y1, y2)))
        
        if self.reduce != 1:
            img = cv2.resize(img, (self.sz//self.reduce, self.sz//self.reduce), interpolation = cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        vertices = torch.tensor([x1, x2, y1, y2])

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - self.mean) / self.std), -1, vertices
        else:
            return img2tensor((img / 255.0 - self.mean) / self.std), idx, vertices

net512s = []
for model in glob.glob('checkpoint/*b3*512*.pth'):
    net = smp.Unet(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=1, activation=None)
    net = net.cuda()
    checkpoint = torch.load(model)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net512s.append(net)

def submission():
    names, rles = [], []
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        ds_512 = HubMAPDataset(idx=row['id'], sz=512, reduce=2, mean=mean_512, std=std_512)
        test_loader_512 = DataLoader(ds_512, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)
        
        preds = np.zeros(ds_512.shape, dtype=np.float16)
        for (x, y, vertices) in test_loader_512:
            py = None
            with torch.no_grad():
                if ((y >= 0).sum() > 0):
                    x = x[y >= 0].to(device)
                    vertices = vertices[y >= 0]
                    y = y[y >= 0]

                    for model in net512s:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p.detach()
                        else:
                            py += p.detach()      
                    py /= len(net512s)
                    py = py.permute(0, 2, 3, 1)
                    py = py.squeeze(-1).cpu().numpy()

            if py is None: continue
            for j, p in enumerate(py):
                if (py.shape[0] != vertices.shape[0]):
                    print(py.shape)
                    print(vertices.shape)

                p = cv2.resize(p, (1024, 1024))
                x1, x2, y1, y2 = vertices[j]
                preds[x1:x2, y1:y2] += p

        mask = (preds >= 0.4)
        rle = rle_encode_less_memory(mask)

        names.append(row['id'])
        rles.append(rle)
        del preds, mask, ds_512, test_loader_512
        gc.collect()

    df = pd.DataFrame({'id': names, 'predicted': rles})
    df.to_csv('submission.csv',index=False)
    return df

df = submission()
