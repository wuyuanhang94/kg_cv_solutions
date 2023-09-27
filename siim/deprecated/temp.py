import pandas as pd


train_df = pd.read_csv('/raid/yiw/siim/train_df.csv')
train_df['path'] = train_df['path'].apply(lambda x: '/raid/yiw/siim/input/' + x[23:])
train_df.to_csv('train_df.csv', index=False)

# train_df = pd.read_csv('/raid/yiw/siim/input/train_df.csv')
# train_df['path'] = train_df['path'].apply(lambda x: '/raid/yiw/siim/input/' + x[19:])
# train_df.to_csv('train_df.csv', index=False)



# import timm

# timm.list_models()

# import torch
# import glob

# for m in glob.glob('/raid/yiw/siim/checkpoint/*.pth'):
#     c = torch.load(m)
#     print(m, c['auc'])
