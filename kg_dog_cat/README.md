# Dogs & Cats Classification
- Implemented by ResNet18, ResNet50
- ResNet18 from scratch achieved 0.18723(LogLoss), took a lot of time training this from scratch, using transfer learning instead
- ResNet50 from scratch achieved 0.16305(LogLoss)
- ResNet50 freeze and train FC achieved 0.09747(LogLoss), much faster

一般分类网络降采样32倍
resnet conv1 stride是2，紧接着一个maxpool

resnet block总共4个block 一般第一个stride是2

resnet 每个layer的block都有shortcut，但并不是每个网络都downsample


