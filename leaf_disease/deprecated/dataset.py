import os
import tqdm
import torch
import random
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

cwd = os.path.abspath(os.path.dirname(__file__))
dPath = os.path.abspath(os.path.join(cwd, 'input'))

class LeafDataset(Dataset):
    def __init__(self, data_path, label_path, train, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.path_list = os.listdir(data_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        df = pd.read_csv(self.label_path)
        if self.train:
            label = df.loc[df['image_id'] == img_path, 'label'].item()
        else:
            label = img_path.split('.')[0]
        label = torch.as_tensor(label, dtype=torch.int64)
        img_path = os.path.join(self.data_path, img_path) #完整路径
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.path_list)

def test():
    train_set = LeafDataset(data_path=os.path.join(dPath, 'train_images'), label_path=os.path.join(dPath, 'train.csv'), train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
    print('train batch:', next(iter(train_loader))[0].shape)

if __name__ == '__main__':
    test()