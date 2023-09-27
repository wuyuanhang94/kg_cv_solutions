import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from tqdm import tqdm
import skimage.io

cwd = os.path.abspath('.')
data_path = os.path.join(cwd, 'input')
train_path = os.path.join(data_path, 'train')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')
info_path = os.path.join(data_path, 'HuBMAP-20-dataset_information.csv')

label_df = pd.read_csv(label_path)
info_df = pd.read_csv(info_path)
test_df = pd.read_csv(csv_path)
DEBUG = 1

tile_size = 512
ext = 'png'

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

    plt.imshow(image)
    plt.imshow(mask, alpha=0.4)
    plt.show()

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def make_tiles(split='train'):
    os.makedirs(f'{split}_tiles/images', exist_ok=True)
    os.makedirs(f'{split}_tiles/masks', exist_ok=True)

    if split == 'train':
        df = label_df
    else:
        df = test_df
    
    info_dict = {}
    info_dict['images'] = []
    info_dict['masks'] = []

    for idx in range(df.shape[0]):
        img_id = df.id[idx]
        path = f"{data_path}/{split}/{img_id}.tiff"
        print(path)
        img = skimage.io.imread(path).squeeze()
        mask = rle2mask(df.encoding[idx], shape=img.shape[1::-1])

        x_max, y_max = img.shape[:2]

        for x0 in tqdm(range(0, x_max, tile_size)):
            x1 = min(x_max, x0+tile_size)
            for y0 in range(0, y_max, tile_size):
                y1 = min(y_max, y0+tile_size)

                img_tile = img[x0:x1, y0:y1]
                mask_tile = mask[x0:x1, y0:y1]

                img_tile_name = f"{img_id}_{x0}-{x1}x_{y0}-{y1}y.{ext}"
                img_tile_path = f"../{split}_tiles/images/{img_tile_name}"
                mask_tile_name = f"{img_id}_{x0}-{x1}x_{y0}-{y1}y.png"
                mask_tile_path = f"../{split}_tiles/masks/{mask_tile_name}"

                cv2.imwrite(img_tile_path, cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_tile_path, mask_tile)

                info_dict['images'].append(img_tile_name)
                info_dict['masks'].append(mask_tile_name)
        
        info_df = pd.DataFrame(data=info_dict)
        info_df = info_df.sort_values('images')
        info_df.to_csv('info.csv', index=None)

make_tiles()