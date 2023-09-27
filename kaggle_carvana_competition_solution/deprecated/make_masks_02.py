import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image
from tqdm import tqdm

def rle_encode(mask_rle, shape=(1280, 1918)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    # starts: 879386, 881253, 883140, ...
    # lengths: 40, 141, 205, ...
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255 # 如果要可视化 就255
        # img[lo:hi] = 1
    return img.reshape(shape) # reshape成1280行 1918列

# bPath = '/datadisk/kg/car/input'
bPath = '/datadisk/kg/carvana/kaggle_carvana_competition_solution_pytorch/input'
maskPath = os.path.join(bPath, 'train_masks_own')
if not os.path.isdir(maskPath):
    os.mkdir(maskPath)
rle_csv = pd.read_csv(os.path.join(bPath, 'train_masks.csv'))
imgNames = [img.split('.')[0] for img in rle_csv['img']]

for name, rle in tqdm(zip(imgNames, rle_csv['rle_mask'])):
    mask_img = rle_encode(rle)
    mask_img_name = os.path.join(maskPath, name+'_mask.jpg')
    # Image.fromarray(mask_img).save(mask_img_name)
    cv2.imwrite(mask_img_name, mask_img)
