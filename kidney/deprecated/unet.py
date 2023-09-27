import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

unet = smp.Unet()

class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1)
                )
                for in_ch, out_ch in zip(input_channels, output_channels)
            ]
        )
    
    def forward(self, xs:list, last_layer):
        hcs = [
            F.interpolate(c(x), scale_factor=2**(len(self.convs)-i), mode='bilinear')
            for i, (c, x) in enumerate(zip(self.convs, xs))
        ]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class UnetBlock(nn.Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf:int=None, blur:bool=False, self_attention:bool=False, **kwargs):
        super().__init__()
        self

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6,12,18,24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU()
        )
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(
            m.conv1,
            m.bn1,
            nn.ReLU(inplace=True)
        )
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
            m.layer1
        )
        self.enc2 = m.layer2
        self.enc3 = m.layer3
        self.enc4 = m.layer4

        self.aspp = None

