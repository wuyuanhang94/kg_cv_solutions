import torch
import glob

for net in glob.glob('checkpoint/*.pth'):
    checkpoint = torch.load(net)
    print(net, checkpoint['acc'], checkpoint['loss'])