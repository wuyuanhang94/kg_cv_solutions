import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=2):
        super(Block, self).__init__()

        group_width = cardinality * bottleneck_width

        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNext(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes):
        super(ResNext, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)
        # 16 = block.expansion的4次方
        self.linear = nn.Linear(cardinality*bottleneck_width*16, num_classes)
    
    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNext50_32x4d(num_classes=100):
    return ResNext(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

def ResNext101_32x4d(num_classes=100):
    return ResNext(num_blocks=[3, 4, 23, 3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

def test():
    net = ResNext101_32x4d()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)

if __name__ == '__main__':
    test()