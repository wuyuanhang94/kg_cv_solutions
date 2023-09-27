import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
import tifffile
import tifffile as tiff

data_path = os.path.abspath(os.path.join(os.path.curdir, 'input/hubmap-kidney-segmentation'))
train_path = os.path.join(data_path, 'train')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')
info_path = os.path.join(data_path, 'HuBMAP-20-dataset_information.csv')
cwd = os.path.abspath('.')

label_df = pd.read_csv(label_path)
info_df = pd.read_csv(info_path)
DEBUG = 1

def read_image(image_id, scale=None, verbose=1):
    image = tifffile.imread(
        os.path.join(data_path, f"train/{image_id}.tiff")
    )
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    
    mask = rle_decode(
        label_df[label_df["id"] == image_id]["encoding"].values[0], 
        (image.shape[1], image.shape[0])
    )
    
    if verbose:
        print(f"[{image_id}] Image shape: {image.shape}")
        print(f"[{image_id}] Mask shape: {mask.shape}")
    
    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        mask = cv2.resize(mask, new_size)
        
        if verbose:
            print(f"[{image_id}] Resized Image shape: {image.shape}")
            print(f"[{image_id}] Resized Mask shape: {mask.shape}")
        
    return image, mask

def read_tiff(image_file):
    image = tiff.imread(image_file)
    print(image.shape)
    # (1, 1, 3, 16180, 27020) 或者 其它
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    return image

def rle_decode(mask_rle, shape):
    '''
    mask_rle 格式:
        starts: 7464094, 7480273, 7496453, ...
        lengths: 59, 64, 205, ...
    注意: 它是按列flatten的，这是flatten之后的index，所以注意height和width以及是否需要transpose
    '''
    s = mask_rle.split(' ')
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def run_check_tile():
    id = '0486052bb'
    image_file = os.path.join(train_path, f'{id}.tiff')
    image = read_tiff(image_file)
    height, width = image.shape[:2]
    print(height, width)

    encoding = label_df.loc[label_df.id == id,'encoding'].values[0]
    width_pixels = info_df.loc[info_df.image_file == '0486052bb.tiff', 'width_pixels'].values[0]
    height_pixels = info_df.loc[info_df.image_file == '0486052bb.tiff', 'height_pixels'].values[0]
    print(height_pixels, width_pixels)
    mask = rle_decode(encoding, (width_pixels, height_pixels))
    if DEBUG:
        # plt.figure()
        # plt.imshow(image)
        # plt.imshow(mask, alpha=0.4)
        # plt.show()
        cv2.imshow('image', image)
        cv2.waitkey(0)

def plot_image_and_mask(image, mask, image_id):
    plt.figure(figsize=(16, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image {image_id}", fontsize=18)
    
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(mask, cmap="hot", alpha=0.5)
    plt.title(f"Image {image_id} + mask", fontsize=18)    
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="hot")
    plt.title(f"Mask", fontsize=18)    
    
    plt.show()

run_check_tile()
# image_id = "0486052bb"
# image, mask = read_image(image_id, 2)
# plot_image_and_mask(image, mask, image_id)

