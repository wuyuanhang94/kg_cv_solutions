import segmentation_models_pytorch as smp
import torch

unet = smp.Unet(encoder_name='resnet34')
x = torch.rand(2, 3, 512, 512)
y = unet(x)
y