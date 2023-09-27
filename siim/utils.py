import torch
import glob

models = glob.glob('../input/siim-study-models/*.pth')
models.sort(key=lambda x: x.split('/')[-1].split('-')[-2][-1])

perf = {}

for m in models:
    c = torch.load(m)
    perf[m] = c['auc']

p = sorted(perf.items(), key=lambda x: x[1])
for k, v in p:
    print(k.ljust(80), v)