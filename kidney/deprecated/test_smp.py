import segmentation_models_pytorch as smp
import torch

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=1,                 # define number of output labels
)
model = smp.Unet(classes=1, activation='sigmoid', aux_params=aux_params)
# print(model)

x = torch.randn(5, 3, 256, 256)
y = model(x)
print(y)
print(y[0].shape)