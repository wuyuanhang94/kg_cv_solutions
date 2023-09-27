import os
import torch
import glob

for net in glob.glob('stage3/*tf_eff*.pth'):
    checkpoint = torch.load(net)
    print(net, checkpoint['auc'])
