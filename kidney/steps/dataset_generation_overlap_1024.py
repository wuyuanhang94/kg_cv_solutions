import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
from torch.utils import data
from tqdm.notebook import tqdm
import zipfile
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import torch

sz = 1024
reduce = 1
min_overlap = 384
s_th = 40
p_th = 1000 * (sz//256) ** 2

MASKS       = 'input/train.csv'
DATA        = 'input/train'
OUT_TRAIN   = 'hubmap1024/train.zip'
OUT_MASKS   = 'hubmap1024/masks.zip'

df_masks = pd.read_csv(MASKS).set_index('id')

'''
Note:
1. overlap
2. wo padding
'''

def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

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

class HubMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce, encs=None):
        self.data = rasterio.open(os.path.join(DATA, idx+'.tiff'), num_threads=12)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce * sz # fixed 1024
        self.make_grid = make_grid(self.shape, window=self.sz, min_overlap=min_overlap)
        self.mask = enc2mask(encs, (self.shape[1], self.shape[0])) if encs is not None else None

    def __len__(self):
        return len(self.make_grid)
    
    def __getitem__(self, idx):
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask = np.zeros((self.sz, self.sz), np.uint8)

        x1, x2, y1, y2 = self.make_grid[idx]
        if self.data.count == 3:
            img = self.data.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
            img = np.moveaxis(img, 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[:, :, i] = layer.read(1, window=Window.from_slices((x1, x2), (y1, y2)))

        if self.mask is not None:
            mask = self.mask[x1:x2, y1:y2]
        
        if self.reduce != 1:
            img = cv2.resize(img, (self.sz//reduce, self.sz//reduce), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.sz//reduce, self.sz//reduce), interpolation = cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        vertices = torch.tensor([x1, x2, y1, y2])

        return img, mask, vertices, (-1 if (s > s_th).sum() <= p_th or img.sum() <= p_th else idx)

x_tot, x2_tot = [], []
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,\
    zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for index, encs in tqdm(df_masks.iterrows(), total=len(df_masks)):
        ds = HubMAPDataset(index, encs=encs)
        for i in range(len(ds)):
            im, m, vertices, idx = ds[i]
            if idx < 0: continue
            x_tot.append((im/255.0).reshape(-1, 3).mean(0))
            x2_tot.append(((im/255.0)**2).reshape(-1, 3).mean(0))

            im = cv2.imencode('.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{index}_{idx:04d}.png', im)
            m = cv2.imencode('.png', m)[1]
            mask_out.writestr(f'{index}_{idx:04d}.png', m)

img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)

'''
plotting
'''

columns, rows = 12, 12
idx0 = 0
fig = plt.figure(figsize=(columns*4, rows*4))
with zipfile.ZipFile(OUT_TRAIN, 'r') as img_arch,\
    zipfile.ZipFile(OUT_MASKS, 'r') as msk_arch:
    fnames = sorted(img_arch.namelist())[8:]
    for i in range(rows):
        for j in range(columns):
            idx = i + j*columns
            img = cv2.imdecode(np.frombuffer(img_arch.read(fnames[idx0+idx]), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imdecode(np.frombuffer(msk_arch.read(fnames[idx0+idx]), 
                                              np.uint8), cv2.IMREAD_GRAYSCALE)
    
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('on')
            plt.imshow(Image.fromarray(img))
            plt.imshow(Image.fromarray(mask), alpha=0.3)
plt.show()
