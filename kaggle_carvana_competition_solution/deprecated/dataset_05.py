import os
import sys
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from train_transform_04 import TrainTransform
import torchvision.transforms as transforms

# 1. use PIL Image rather than cv2.imread
# 2. masks should not normalize

# data_transform = {
#     "train": {
#         # 即便是把images 和 masks分开仍然是错的 因为images有什么样的transform mask 也有什么样的transform 而不应该是两边自己random自己的
#         # 要把images 和 masks 放在同一个transform里
#         "images":
#             transforms.Compose([
#                 transforms.Resize((512, 512)),
#                 transforms.RandomResizedCrop(512),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.6982, 0.6909, 0.6840], [0.2327, 0.2369, 0.2345])]),
#         "masks":
#             transforms.Compose([
#                 transforms.Resize((512, 512)),
#                 transforms.RandomResizedCrop(512),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor()]),
#     },
#     "val": transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.6982, 0.6909, 0.6840], [0.2327, 0.2369, 0.2345])
#     ])
# }

class CarvanaDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.images = os.listdir(images_path)
        self.masks_path = masks_path
        self.masks = os.listdir(masks_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.images[idx])
        car_img = Image.open(img_path)
        img_id = self.images[idx].split('.')[0]
        mask_path = os.path.join(self.masks_path, img_id+'_mask.jpg')
        car_mask = Image.open(mask_path)
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 20))
        # axes[0].imshow(car_img)
        # axes[1].imshow(car_mask)
        # plt.show()
        if self.transform != None:
            car_img, car_mask = self.transform(car_img, car_mask)
        return car_img, car_mask

def test():
    train_set = CarvanaDataset(images_path='/datadisk/kg/carvana/kaggle_carvana_competition_solution_pytorch/input/train',
                               masks_path='/datadisk/kg/carvana/kaggle_carvana_competition_solution_pytorch/input/train_masks',
                               transform=TrainTransform(new_size=(512, 512)))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=4)
    sample_images, sample_masks = next(iter(train_loader))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(30, 50))
    for i in range(4):
        axes[0][i].imshow(transforms.ToPILImage()(sample_images[i, ...]).convert('RGB'))
    for i in range(4):
        axes[1][i].imshow(transforms.ToPILImage()(sample_masks[i, ...]).convert('L'))
    plt.show()

def splitDataset():
    pass
    # full_dataset = MyDataset(data_path='/home/yiw/daily/dog_cat_titan/input/train', train=True)
    # train_size = int(0.8 * len(full_dataset))
    # validate_size = int(0.2 * len(full_dataset))

    # train_set, validate_set = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4)
    # validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=2, num_workers=4)

if __name__ == '__main__':
    test()
