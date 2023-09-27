import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import os
import argparse

from model.resnet import *
from model.googlenet import *

from dataset import MyDataset

# logloss 提交是狗的probability
def submission(csv_path, test_loader, device, net):
    result_dict = {}
    result_dict['id'] = []
    result_dict['label'] = []
    net = net.to(device)
    with torch.no_grad():
        for _, (inputs, ids) in enumerate(test_loader):
            inputs, ids = inputs.to(device), ids.to(device)

            outputs = net(inputs)
            softmax_func = nn.Softmax(dim=1)
            softmax_value = softmax_func(outputs)
            for i in range(len(softmax_value)):
                result_dict['id'].append(ids[i].item())
                result_dict['label'].append(softmax_value[i][1].item())
    result_df = pd.DataFrame(data=result_dict)
    result_df = result_df.sort_values("id")
    result_df.to_csv(csv_path, index=None)
    return result_df

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dataset = MyDataset(data_path='./input/test1/test', train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=4)

net = ResNet50(num_classes=2)
net = net.to(device)
# 前面train的时候用了cudnn 这里也要是cudnn的格式 否则报错
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/dog-cat-resnet50.pth')
net.load_state_dict(checkpoint['net'])

df = submission('./sampleSubmission.csv', test_loader, device, net)
print("submission.csv generated.")