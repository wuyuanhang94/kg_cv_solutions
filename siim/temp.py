import pandas as pd

train_df = pd.read_csv("/home/yiw/siim/train_df.csv")
train_df['path'] = train_df['path'].apply(lambda x: '/home/yiw/siim/' + x[12:])
train_df.to_csv('train_df.csv', index=False)

# import timm

# timm.list_models()

# import torch
# import glob

# for m in glob.glob('/raid/yiw/siim/checkpoint/*.pth'):
#     c = torch.load(m)
#     print(m, c['auc'])
