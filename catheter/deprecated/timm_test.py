import timm

effb4 = timm.create_model('tf_efficientnet_b4', pretrained=True)
print(effb4)

resnet200d = timm.create_model('resnet200d', pretrained=True)
