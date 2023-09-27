import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random

def rle_encode(mask_rle, shape=(1280, 1918)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    # starts: 879386, 881253, 883140, ...
    # lengths: 40, 141, 205, ...
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape) # reshape成1280行 1918列

bPath = '/datadisk/kg/carvana/kaggle_carvana_competition_solution_pytorch/input'

rle_csv = pd.read_csv(os.path.join(bPath, 'train_masks.csv'))
'''
print(rle_csv.head())
                   img                                           rle_mask
0  00087a6bd4dc_01.jpg  879386 40 881253 141 883140 205 885009 17 8850...
1  00087a6bd4dc_02.jpg  873779 4 875695 7 877612 9 879528 12 881267 15...
2  00087a6bd4dc_03.jpg  864300 9 866217 13 868134 15 870051 16 871969 ...
3  00087a6bd4dc_04.jpg  879735 20 881650 26 883315 92 883564 30 885208...
4  00087a6bd4dc_05.jpg  883365 74 883638 28 885262 119 885550 34 88716...
'''
# print(rle_csv['img'])
imgs = rle_csv['img']
# rle_mask都是展平之后得到的 而且这里的rle 是 起始位置：个数 的形式
rle_masks = rle_csv['rle_mask']
# print(rle_masks['00087a6bd4dc_01.jpg'])
# print(rle_masks.loc[0])

idx = random.randint(0, len(imgs))
si, sm = imgs[idx], rle_masks[idx]
si = os.path.join(bPath, 'train', si)

fig, axes = plt.subplots(1, 3, figsize=(15, 40))
axes[0].imshow(cv2.imread(si))
axes[1].imshow(rle_encode(sm))
axes[2].imshow(cv2.imread(si))
axes[2].imshow(rle_encode(sm), alpha=0.4)
plt.show()