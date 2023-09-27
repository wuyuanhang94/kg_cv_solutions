import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Down, self).__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Up, self).__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)
	
	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes

		self.input = DoubleConv(n_channels, 64)
		
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 1024)
		self.up1 = Up(1024, 512)
		self.up2 = Up(512, 256)
		self.up3 = Up(256, 128)
		self.up4 = Up(128, 64)

		self.output = nn.Conv2d(64, n_classes, kernel_size=1)

	def forward(self, x):
		x1 = self.input(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.output(x)
		return logits
