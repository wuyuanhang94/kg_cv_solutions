import os
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

data_dir = os.path.join(os.getcwd(), 'input')
print(data_dir)

def get_mean_and_std(dataset, batch_size=8):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std")
    for inputs, _ in tqdm(dataloader):
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_((len(dataset)/batch_size))
    std.div_((len(dataset)/batch_size))
    return mean, std

transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
train_test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transforms)

print(get_mean_and_std(train_test_dataset))
# (tensor([0.4882, 0.4557, 0.4172]), tensor([0.2545, 0.2480, 0.2504]))