# cifar10 demo with PyTorch
- Implemented by ResNet18, ResNet50 and GoogLenet
- GoogLenet has a faster training speed than ResNet50, but achieving similar classification precision finally, about 85% after 100 epochs
## Results
```
(pth) yiw@iNvidian:~/daily/cifar-demo$ python main.py -r
==> Preparing data...
Files already downloaded and verified
Files already downloaded and verified
==> Building model...
GoogLenet(
  (pre_layers): Sequential(
    (0): Conv2d(3, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (a3): Inception(
    (b1): Sequential(
      (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (b3): Inception(
    (b1): Sequential(
      (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (a4): Inception(
    (b1): Sequential(
      (0): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (b4): Inception(
    (b1): Sequential(
      (0): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (c4): Inception(
    (b1): Sequential(
      (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (d4): Inception(
    (b1): Sequential(
      (0): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (e4): Inception(
    (b1): Sequential(
      (0): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (a5): Inception(
    (b1): Sequential(
      (0): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (b5): Inception(
    (b1): Sequential(
      (0): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (b2): Sequential(
      (0): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
    (b3): Sequential(
      (0): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
    )
    (b4): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
    )
  )
  (avgpool): AvgPool2d(kernel_size=8, stride=1, padding=0)
  (linear): Linear(in_features=1024, out_features=10, bias=True)
)
==> Resuming from checkpoint...

Epoch: 32
 [=========================== 391/391 ============================>]  Step: 1s614ms | Tot: 1m28s | Loss: 0.307 | Acc: 89.386% (44693/50000)                                                                        
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s148ms | Loss: 0.744 | Acc: 77.170% (7717/10000)                                                                          

Epoch: 33
 [=========================== 391/391 ============================>]  Step: 152ms | Tot: 1m26s | Loss: 0.308 | Acc: 89.374% (44687/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s189ms | Loss: 0.725 | Acc: 76.980% (7698/10000)                                                                          

Epoch: 34
 [=========================== 391/391 ============================>]  Step: 141ms | Tot: 1m25s | Loss: 0.308 | Acc: 89.232% (44616/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s218ms | Loss: 0.690 | Acc: 77.430% (7743/10000)                                                                          

Epoch: 35
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m26s | Loss: 0.299 | Acc: 89.812% (44906/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 54ms | Tot: 5s275ms | Loss: 0.943 | Acc: 70.230% (7023/10000)                                                                          

Epoch: 36
 [=========================== 391/391 ============================>]  Step: 156ms | Tot: 1m26s | Loss: 0.304 | Acc: 89.544% (44772/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 52ms | Tot: 5s247ms | Loss: 0.605 | Acc: 79.480% (7948/10000)                                                                          

Epoch: 37
 [=========================== 391/391 ============================>]  Step: 149ms | Tot: 1m27s | Loss: 0.300 | Acc: 89.608% (44804/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s340ms | Loss: 0.623 | Acc: 80.360% (8036/10000)                                                                          

Epoch: 38
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m27s | Loss: 0.302 | Acc: 89.606% (44803/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 56ms | Tot: 5s457ms | Loss: 0.636 | Acc: 80.900% (8090/10000)                                                                          

Epoch: 39
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m25s | Loss: 0.298 | Acc: 89.940% (44970/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s149ms | Loss: 0.455 | Acc: 84.520% (8452/10000)                                                                          

Epoch: 40
 [=========================== 391/391 ============================>]  Step: 141ms | Tot: 1m25s | Loss: 0.299 | Acc: 89.650% (44825/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s181ms | Loss: 0.687 | Acc: 79.030% (7903/10000)                                                                          

Epoch: 41
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m27s | Loss: 0.298 | Acc: 89.742% (44871/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 59ms | Tot: 5s500ms | Loss: 0.818 | Acc: 75.700% (7570/10000)                                                                          

Epoch: 42
 [=========================== 391/391 ============================>]  Step: 141ms | Tot: 1m27s | Loss: 0.295 | Acc: 90.030% (45015/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s190ms | Loss: 0.591 | Acc: 80.890% (8089/10000)                                                                          

Epoch: 43
 [=========================== 391/391 ============================>]  Step: 157ms | Tot: 1m29s | Loss: 0.295 | Acc: 89.806% (44903/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 55ms | Tot: 5s577ms | Loss: 0.736 | Acc: 77.590% (7759/10000)                                                                          

Epoch: 44
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m28s | Loss: 0.296 | Acc: 89.874% (44937/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s167ms | Loss: 0.600 | Acc: 80.780% (8078/10000)                                                                          

Epoch: 45
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m26s | Loss: 0.291 | Acc: 89.948% (44974/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s401ms | Loss: 0.440 | Acc: 85.510% (8551/10000)                                                                          
Saving...

Epoch: 46
 [=========================== 391/391 ============================>]  Step: 146ms | Tot: 1m28s | Loss: 0.289 | Acc: 89.936% (44968/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 52ms | Tot: 5s319ms | Loss: 0.630 | Acc: 79.500% (7950/10000)                                                                          

Epoch: 47
 [=========================== 391/391 ============================>]  Step: 142ms | Tot: 1m27s | Loss: 0.290 | Acc: 90.008% (45004/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 51ms | Tot: 5s260ms | Loss: 0.575 | Acc: 81.030% (8103/10000)                                                                          

Epoch: 48
 [=========================== 391/391 ============================>]  Step: 152ms | Tot: 1m27s | Loss: 0.292 | Acc: 89.926% (44963/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 54ms | Tot: 5s307ms | Loss: 2.252 | Acc: 57.220% (5722/10000)                                                                          

Epoch: 49
 [=========================== 391/391 ============================>]  Step: 144ms | Tot: 1m27s | Loss: 0.295 | Acc: 89.658% (44829/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 54ms | Tot: 5s474ms | Loss: 0.395 | Acc: 86.590% (8659/10000)                                                                          
Saving...

Epoch: 50
 [=========================== 391/391 ============================>]  Step: 144ms | Tot: 1m27s | Loss: 0.289 | Acc: 89.992% (44996/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 52ms | Tot: 5s297ms | Loss: 0.819 | Acc: 75.680% (7568/10000)                                                                          

Epoch: 51
 [=========================== 391/391 ============================>]  Step: 160ms | Tot: 1m29s | Loss: 0.289 | Acc: 90.002% (45001/50000)                                                                          
 [=========================== 100/100 ============================>]  Step: 56ms | Tot: 5s629ms | Loss: 1.042 | Acc: 69.750% (6975/10000)  
```
