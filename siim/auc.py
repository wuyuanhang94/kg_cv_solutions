import torch
import glob

models = glob.glob('/home/yiw/siim/checkpoint/*pth')
models.sort(key=lambda x: x.split('/')[-1].split('-')[-2][-1])

perf = dict()

for m in models:
    c = torch.load(m)
    # print(m.ljust(80), c['auc'])
    perf[m] = c['auc']

perf_s = sorted(perf.items(), key=lambda x: x[1])
for k, v in perf_s:
    print(k.ljust(80), v)
