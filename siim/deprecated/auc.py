import torch
import glob

models = glob.glob('/raid/yiw/siim/checkpoint/*pth')
models.sort(key=lambda x: x.split('/')[-1].split('-')[-2][-1])

for m in models:
    c = torch.load(m)
    print(m.ljust(80), c['auc'])

