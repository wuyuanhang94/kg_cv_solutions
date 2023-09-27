import os
import tqdm
import random
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

from efficientnet_b6 import *

data_path = os.path.abspath(os.path.join(os.path.pardir, 'input/cassava-leaf-disease-classification'))
train_path = os.path.join(data_path, 'train_images')
label_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test_images')
csv_path = os.path.join(data_path, 'sample_submission.csv')
cwd = os.path.abspath('.')

dataset_transforms = {
    'train': transforms.Compose([
                    transforms.Resize((528, 528)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.RandomCrop((528, 528), padding=3),
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4304, 0.4968, 0.3135], [0.2358, 0.2387, 0.2256])
                ]),
    'other': transforms.Compose([
                    transforms.Resize((528, 528)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4304, 0.4968, 0.3135], [0.2358, 0.2387, 0.2256])
                ])
}

class CassavaDataset(Dataset):
    def __init__(self, df, dPath=None, transforms=None, test=False):
        super(CassavaDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.dPath = dPath
        self.transforms = transforms
        self.test = test
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_name = self.df['image_id'].values[idx]
        img_path = os.path.join(self.dPath, img_name)
        img = Image.open(img_path)
        img = self.transforms(img)
        if self.test:
            label = int(img_name.split('.')[0])
        else:
            label = self.df['label'].values[idx]
        return img, label

def submission(csv_path, test_loader, device, net):
    result_dict = {}
    result_dict['image_id'] = []
    result_dict['label'] = []
    net = net.to(device)
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs, ids = inputs.to(device), ids.to(device)

            outputs = net(inputs)
            # print(outputs.softmax(dim=1))
            preds = outputs.argmax(dim=1)
            for i in range(len(preds)):
                result_dict['image_id'].append(f'{ids[i].item()}.jpg')
                result_dict['label'].append(preds[i].item())
    result_df = pd.DataFrame(data=result_dict)
    result_df = result_df.sort_values('image_id')
    submission_path = 'submission.csv'
    result_df.to_csv(submission_path, index=None)
    return result_df

device = torch.device('cuda')
test_df = pd.read_csv(csv_path)
test_set = CassavaDataset(test_df, test_path, dataset_transforms['other'], True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, num_workers=1)

net = EfficientNet.from_name('efficientnet-b6').to(device)
net._fc = nn.Linear(net._fc.in_features, 5)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
checkpoint = torch.load('../input/efficientnetb6pretrained/leaf-b6.pth')
net.load_state_dict(checkpoint['net'])

df = submission(csv_path, test_loader, device, net)
df