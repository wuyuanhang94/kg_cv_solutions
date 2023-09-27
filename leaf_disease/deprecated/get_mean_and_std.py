import os
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

data_dir = os.path.join(os.path.pardir, 'input')
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
    transforms.Resize(384),
    transforms.ToTensor()
])
train_test_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transforms)

print(get_mean_and_std(train_test_dataset))
# use the value if training from scratch, otherwise use imagenet's result
# scratch   ->  (tensor([0.4304, 0.4968, 0.3135]), tensor([0.2358, 0.2387, 0.2256]))
# pretrained -> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# 384: (tensor([0.4304, 0.4968, 0.3135]), tensor([0.2269, 0.2299, 0.2166]))