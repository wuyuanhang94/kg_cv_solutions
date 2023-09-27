import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> computing mean and std')
    for inputs, _ in tqdm(dataloader):
        for i in range(3):#3个通道
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

data_dir = '/datadisk/kg/carvana/kaggle_carvana_competition_solution_pytorch/input/imagefolder'
train_transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(data_dir, transform=train_transform)

print(get_mean_and_std(trainset))
# (tensor([0.6982, 0.6909, 0.6840]), tensor([0.2327, 0.2369, 0.2345]))