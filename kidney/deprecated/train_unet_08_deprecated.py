import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
from tqdm import tqdm

from dataset_05 import KidneyDataset
from train_transform_04 import TrainTransform
import segmentation_models_pytorch as smp
from losses_07 import *
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = float('inf')
start_epoch = 0

print("==> Preparing data...")
cwd = os.path.abspath('.')
full_dataset = KidneyDataset(images_path=os.path.join(cwd, 'train_tiles/images'),
                               masks_path=os.path.join(cwd, 'train_tiles/masks'),
                               transform=TrainTransform(new_size=(512, 512)))
train_size = int(0.8 * len(full_dataset))
validate_size = len(full_dataset) - train_size

train_set, validate_set = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
train_batchsize = 2
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
val_batchsize = 8
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=4, num_workers=2)

# Model
print("==> Building model...")
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=1,                 # define number of output labels
)
net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid', aux_params=aux_params)
# checkpoint = torch.load('checkpoint/kidney-unet.pth')
# net.load_state_dict(checkpoint['net'])
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9, weight_decay=1e-6)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, masks, labels) in progress_bar:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        
        optimizer.zero_grad()
        y_pred, logits = net(inputs)
        loss = nn.BCELoss(logits, labels)
        if max(labels) == 1:
            seg_idx = labels == 1
            loss += dice_loss(y_pred[seg_idx, ...], masks[seg_idx, ...])
        # 累加分类loss 和 分割loss
        if max(labels) == 0: # pure classification
            criterion = nn.BCELoss()
            _, logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_description('Loss: %.3f | Dice: %.3f' % 
                                        (train_loss / (batch_idx+1), 0))
        elif min(labels) == 1: # pure segmentation
            criterion = bce_dice_loss
            y_pred, logits = net(inputs)
            loss = criterion(y_pred, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            dice = dice_ceff(y_pred, masks)
            progress_bar.set_description('Loss: %.3f | Dice: %.3f' % 
                                    (train_loss / (batch_idx+1), dice))
        else: # mixed
            cls_idx = labels == 0
            seg_idx = labels == 1
            masks = masks.to(device, dtype=torch.float32)
            y_pred, logits = net(inputs)
            
            loss = bce_dice_loss(y_pred[seg_idx, ...], masks[seg_idx, ...])
            loss += nn.BCELoss()(logits[cls_idx, ...], labels[cls_idx, ...])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            dice = dice_ceff(y_pred[seg_idx, ...], masks[seg_idx, ...])
            progress_bar.set_description('Loss: %.3f | Dice: %.3f' % 
                                    (train_loss / (batch_idx+1), dice))
            if dice > 0.99:
                os.makedirs('checkpoint', exist_ok=True)
                print('Saving model...')
                state = {
                    'net': net.state_dict(),
                    'dice': dice,
                    'epoch': epoch
                }
                torch.save(state, 'checkpoint/kidney-unet.pth')


def validate(epoch):
    net.eval()
    test_loss = 0.
    global best_loss

    with torch.no_grad():
        progress_bar = tqdm(enumerate(validate_loader), total=len(train_loader))
        for batch_idx, (inputs, masks, labels) in progress_bar:
            inputs, masks, labels = inputs.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            
            y_preds, logits = net(inputs)
            loss = nn.BCELoss()(logits, labels) # classification loss
            
            seg_idx = y_preds.max() == 1
            dice = dice_ceff(y_pred[seg_idx, ...], masks[seg_idx, ...])
            loss += (1 - dice)

            test_loss += loss.item()
            progress_bar.set_description('Loss: %.3f | Dice: %.3f' % (test_loss / (batch_idx+1), dice))
            
        if test_loss/(batch_idx+1) < best_loss:
            best_loss = test_loss / (batch_idx+1)
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            print('Saving model...')
            state = {
                'net': net.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }
            torch.save(state, 'checkpoint/kidney-unet.pth')

for epoch in range(start_epoch, start_epoch+50):
    # train(epoch)
    validate(epoch)