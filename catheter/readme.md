导管存在及位置分类 - 多标签二分类问题 - 不是利用x光片判断是否是病人

难点在于理解和利用好annotations，有17999份样本包含annotations，剩下12084份样本没有annotations, stage1只利用这17999份样本

为什么resnet family更好？
imagenet state-of-the-art收益来自多尺度，但medical image尺度往往是单一的，变化不大，所以efficientnet的效果在medical image dataset中难以发挥

思路：
- effb7 + resnet200d(512, 700两种图片尺寸融合) + inceptionV3 + densenet121
- teacher-student 训练 annotated + pure images
- BCE + focal loss
- effb7 没有pretrained，使用effb5
- 至少要有两个epoch是light aug？ 冻结BN层

In ensembles the models must have **a high disagreement ratio**, while each base model must be as much accurate as possible. In order to build base models with high disagreement ratio, you can do some of the things bellow:

- [1] Mix diferent models like ResNet and Eff
- [2] Mix models trained on diferent number of epochs (e.g one model - close to overfitting, and another close to bias)
- [3] Mix diferent trained folds models, eg 5 models trained on a 5 - fold cyrcle.
- [4] Mix diferent trained seeds models (models trained on diferent - initial random states)
- [5] Mix models trained on diferent images resolutions.
- [6] TTA (Test Time Augmentation)

soft voting & hard voting
averaging
voting
的优缺点及适用场景

lesson learned: add **inceptionV3** and **densenet121** to your ensembles

- 直接使用AUC作为loss function
- img size 500-700？

diversity 是模型融合的关键 也是leaf 1st的关键

```
for module in model.modules():
    # print(module)
    if isinstance(module, nn.BatchNorm2d):
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(False)
        module.eval()
```
