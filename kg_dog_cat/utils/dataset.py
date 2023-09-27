import os
from torchvision.transforms.transforms import RandomAffine, RandomAutocontrast, RandomVerticalFlip
import tqdm
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

cwd = os.path.abspath(os.path.dirname(__file__))
dPath = os.path.abspath(os.path.join(cwd, '../input'))

class MyDataset(Dataset):
    def __init__(self, data_path, train, transform=None):
        self.data_path = data_path
        self.train = train
        self.path_list = os.listdir(data_path)
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(size=(228, 228)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAutocontrast(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4882, 0.4557, 0.4172], [0.2545, 0.2480, 0.2504])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(size=(228, 228)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4882, 0.4557, 0.4172], [0.2545, 0.2480, 0.2504])
                ])
        else:
            self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        if self.train:
            if img_path.split('.')[0] == 'dog':
                label = 1 # 狗的概率
            else:
                label = 0
        else:
            # 对测试集 我们没有label 直接讲id设为label 方便后面生成predict
            label = int(img_path.split('.')[0])
        label = torch.as_tensor(label, dtype=torch.int64)
        img_path = os.path.join(self.data_path, img_path) #完整路径
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.path_list)

def test():
    train_set = MyDataset(data_path=os.path.join(dPath, 'train'), train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
    print('train batch:', next(iter(train_loader))[0].shape)

    test_set = MyDataset(data_path=os.path.join(dPath, 'test'), train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=8)
    print('test batch: ', next(iter(test_loader))[0].shape)

    img_PIL_Tensor = random.choice(test_set)[0]
    new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')

    import matplotlib.pyplot as plt
    plt.imshow(new_img_PIL)
    plt.show()

if __name__ == '__main__':
    test()