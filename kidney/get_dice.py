import torch
import glob

for cp in glob.glob('pretrained/*unet*.pth'):
    checkpoint = torch.load(cp)
    print(cp, checkpoint['dice'], checkpoint['loss'])
