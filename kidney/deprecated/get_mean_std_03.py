import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from tqdm import tqdm
import skimage.io

data_path = '/datadisk/kg/kaggle_kidney/train_tiles/tmp'
tile_size = 512
ext = 'png'

def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> computing mean and std')
    for inputs, _ in tqdm(dataloader):
        for i in range(3):#3个通道
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

train_transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(data_path, transform=train_transform)

print(get_mean_and_std(trainset))
# (tensor([0.5936, 0.4990, 0.6150]), tensor([0.0757, 0.0967, 0.0623]))