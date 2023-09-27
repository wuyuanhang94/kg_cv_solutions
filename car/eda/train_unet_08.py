import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from dataset_05 import CarvanaDataset
from train_transform_04 import TrainTransform
from unet_06 import UNet
from losses_07 import *
from utils_09 import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = float('inf')
start_epoch = 0

print("==> Preparing data...")
cwd = '/datadisk/kg/car'
full_dataset = CarvanaDataset(images_path=os.path.join(cwd, 'input/train'),
                               masks_path=os.path.join(cwd, 'input/train_masks_own'),
                               transform=TrainTransform(new_size=(512, 512)))
train_size = int(0.8 * len(full_dataset))
validate_size = len(full_dataset) - train_size

train_set, validate_set = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=8, num_workers=2)

# Model
print("==> Building model...")
net = UNet(n_channels=3, n_classes=1)
net = net.to(device)
# print(net)

criterion = bce_dice_loss
optimizer = optim.SGD(net.parameters(), lr=0.08, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.
    
    for batch_idx, (inputs, masks) in enumerate(train_loader):

        # import matplotlib.pyplot as plt
        # _, axes = plt.subplots(2, 2, figsize=(30, 30))
        # for i in range(2):
        #     axes[0][i].imshow(transforms.ToPILImage()(inputs[i, ...]).convert('RGB'))
        # for i in range(2):
        #     axes[1][i].imshow(transforms.ToPILImage()(masks[i, ...]).convert('L'))
        # plt.show()

        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        y_pred = net(inputs)
        loss = criterion(y_pred, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Train Loss: %.3f | Train Dice Coeff: %.3f'
                     % (train_loss / (batch_idx+1), dice_ceff(y_pred, masks)))

def validate(epoch):
    net.eval()
    test_loss = 0.
    global best_loss

    with torch.no_grad():
        for batch_idx, (inputs, masks) in enumerate(validate_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            y_pred = net(inputs)
            loss = criterion(y_pred, masks)

            test_loss += loss.item()
            progress_bar(batch_idx, len(validate_loader), 'Val Loss: %.3f | Val Dice Coeff: %.3f'
                     % (test_loss / (batch_idx+1), dice_ceff(y_pred, masks)))
            
            if test_loss / (batch_idx+1) < best_loss:
                best_loss = test_loss / (batch_idx+1)
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                print('Saving mode...')
                state = {
                    'net': net.state_dict(),
                    'loss': test_loss / (batch_idx+1),
                    'epoch': epoch
                }
                torch.save(state, 'checkpoint/carvana-unet.pth')

for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    validate(epoch)