import torch
import torchvision
import torchvision.transforms as transforms

def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std")
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

print('==> Downloading data..')

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transforms.ToTensor())

print(get_mean_and_std(trainset))
# (tensor([0.4914, 0.4822, 0.4465]), tensor([0.2023, 0.1994, 0.2010]))