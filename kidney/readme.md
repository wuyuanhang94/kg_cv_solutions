Efficientnet-Unet，effb4作为encoder，同时使用ASPP和FPN，1024 original size -> 512，先不加分类head，dice loss + BCE loss, 以后尝试lovasz loss (x) 不好用
dice coefficient作为得分标准，最好还是dice loss直接

这里面太多坑了。。
0. tiff尺寸太大，5w×3w，不用切片方法根本没法处理，了解rasterio lib
1. 小心cv2 rgb bgr顺序，小心rasterio.open是否和cv2.open行为一样
2. mask 是[0, 1] 还是[0, 255]，小心pytorch的归一化
3. unet segm head自带的activation和BCEWithLogitsLoss冲突，使用BCELoss()
4. dice loss, sigmoid, 是否需要 > 0.5 处理？ 不需要，你会使它不可导

fix bugs 之后 如何进一步上分？
1. 尝试更先进的loss function： focal, lovasz(这个不行)，尝试combo loss
2. 使用更大的分辨率，512, 1024
3. unet, fpn, linknet, 尝试ensemble， 效果不好
4. 后处理，阈值选取
5. 如何选取合理的Inference threshold，目前是粗暴地选择0.5, 0.39一下子就提分那么多
6. 512 现在的batch_size 太小了，想办法增大batch_size, 512显然是有效果的
7. unet, fpn, linknet, pspnet四种都用，且用三种size，256，512，1024

如何更好的调整dice loss和bce loss的权重？
在训练初期，bce权重必须很大，比如前10个epoch；然后依次加大dice loss weight；但是对于每一个batch，是有很多negative samples的，如何更好的抑制这些false positives，bce loss weight不能太低。

seed2021 | unet_256       |         fpn_256    | unet_512          | fpn_512
-|-|-|-|- 
fold0 | 0.930550448417663 | 0.9249361600875855 | 0.925700853824615 | x
fold1 | 0.916852617636323 | 0.9112432287074625 | 0.913550609722733 | x
fold2 | 0.923742129431142 | 0.9249148690081263 | 0.920757137722783 | x
fold3 | 0.906942626016329 | 0.9230923541879232 | 0.897643540812804 | x
fold4 | 0.935647146937287 | 0.9313161277476653 | 0.932824162789333 | x

seed9419 | unet_512        |    fpn_512         |   unet_b3_512     | linknet_b3_512    | unet_1024_b3
-|-|-|-|-|-
fold0 | 0.9396215847560337 | 0.9251452599252973 | 0.938142194634392 | 0.934492332594735 | 0.9350617980515
fold1 | 0.9329350015946797 | 0.9100476375647953 | 0.934668202485357 | 0.919725626707077 | 0.9291208059820
fold2 | 0.9137506700576621 | 0.8950285949605576 | 0.916113696199782 | 0.877443379544197 | 0.9062424229412
fold3 | 0.9301710911095142 | 0.9050458129495382 | 0.941623037680983 | 0.923282556235790 | 0.9128987620858
fold4 | 0.9367529749870311 | 0.9195329284667969 | 0.942510433197021 | 0.923563735485076 | 0.9231225884321

大分辨率最好用小分辨率作为pretrain model，1024 用 512
```
scp nvidia@10.19.225.134:/nvidia/.yiw/kidney/checkpoint/*unet*512* ./
scp nvidia@10.19.225.134:/nvidia/.yiw/kidney/checkpoint/*unet*b3*512* ./
scp nvidia@10.19.225.134:/nvidia/.yiw/kidney/checkpoint/*linknet* ./
scp yiw@10.19.183.148:/datadisk/kg/kidney/checkpoint/*fpn* ./
scp nvidia@10.19.225.134:/nvidia/.yiw/kidney/*b3*.py ./
```

或许8fold 比较好，每个fold image变多

1. overlap tiles, 好处是a: tile 边缘会出现至少两次，最多四次，Unet在图像边缘分割能力较弱，原因不清楚，看Unet原论文；b: 增大数据集 如果overlap 那inference的时候是否最好也overlap？
2. 更好的采样策略，提供纯净的只带有mask的训练集训练，那么分类的问题怎么办？dice loss + BCE 或许也能很好的handle
3. 预训练和外部数据集

1024 res反而不好的原因应该是 downsample 次数不够，导致大目标检测有问题

mean_256 = np.array([0.65527532, 0.49901106, 0.69247992])
std_256 = np.array([0.25565283, 0.31975344, 0.21533712])

mean_512 = np.array([0.63759809, 0.4716141, 0.68231112])
std_512 = np.array([0.16475244, 0.22850685, 0.14593643])
mean: [0.63990417 0.4734721  0.68480998] , std: [0.16061672 0.22722983 0.14034663]

mean_1024 = np.array([0.63711163, 0.47114376, 0.68181989])
std_1024 = np.array([0.16736293, 0.23051024, 0.14848679])

overlap - mean: [0.63959578 0.47326392 0.68438224] , std: [0.1633776  0.22941419 0.14310585]

pretrained | dice
-|-
f4 | 0.8925447757427509
f3 | 0.8663155931692857
f1 | 0.9040154732190646
f0 | 0.7516377109747666
f2 | 0.8720820793738732

预训练 + overlap 果然效果不错，unet-b3-512, 现在要尝试fpn-b3-512, 也要pretrained

b2 + 256

有个办法是加入d4到train dataset，还有就是加入所有test tiff到train dataset 混合，既然咱们可以选两个，那就挑一个用这种方法的。
