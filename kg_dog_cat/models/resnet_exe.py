import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        pass


class Resnet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000):
        super(Resnet, self).__init__()
        
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.layer1 = self._make_layer(BasicBlock, 64, blocks_num[0])
        self.layer2 = self._make_layer(BasicBlock, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, num_blocks, stride=1):
        downsample = None
        # self.in_channel是每个block的in_channel
        if stride != 1 or self.in_channel != block.expansion * channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, block.expansion*channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * channel)
            )
        layers = []
        # 第一个block stride = stride 特殊处理一下
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, channel))
            self.in_channel = channel * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18(num_classes=2):
    return Resnet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

if __name__ == '__main__':
    net = ResNet18()
    print(net)

    x = torch.randn(8, 3, 256, 256)
    y = net(x)
    print(y.shape)
