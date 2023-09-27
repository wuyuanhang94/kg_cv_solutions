import torch
import torch.nn as nn
import torch.nn.functional as F

# resnet后面和googlenet一样 使用avgpool2d 而不是全连接

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3,
                                stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3,
                            stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.bn2(self.conv2(out))) #这里也可以不加relu
        out = self.bn2(self.conv2(out))
        # shortcut
        # element-wise add 按元素相加
        # 但是经过两层的卷积之后虽然shape可以用padding控制一致，channel却不一致
        # 解决方法就是x也通过1x1卷积调整channel数量
        # [b, ch_in, h, w] vs [b, ch_out, h, w]
        # out = x + out
        out = self.extra(x) + out
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # 紧跟着的是4个blocks
        self.blk1 = ResBlk(64, 128)
        self.blk2 = ResBlk(128, 256)
        self.blk3 = ResBlk(256, 512)
        self.blk4 = ResBlk(512, 1024)

        self.avgpool = nn.AvgPool2d(32, stride=1)
        self.outlayer = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.outlayer(x)

        return x

def main():
    blk = ResBlk(64, 128)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print("Testing ResBlk first...")
    print(out.shape)

    model = ResNet18()
    tmp = torch.randn(2, 3, 32, 32)
    out = model(tmp)
    print('Testing ResNet18...')
    print(tmp.shape)

if __name__ == '__main__':
    main()