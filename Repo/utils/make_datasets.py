from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
import sys

"""Loads the CIFAR-10 Dataset, and splits it into two sections. This allows for reducing dataset size to train models on smaller dataset sizes and analyze model scalability."""
def download_data(size):
    CIFAR10Orig = torchvision.datasets.CIFAR10("/Users/adithyagiri/Desktop/STS/Repo/Datasets/CIFAR-10", train=True, download=True)
    seed = 1
    data_len = 50000
    val_size = 10000
    CIFAR10, CIFAR_Val, Remaining = torch.utils.data.random_split(
            CIFAR10Orig, [size, val_size, data_len-val_size-size], generator=torch.Generator().manual_seed(seed))

    print(len(CIFAR10))
    CIFAR_Train = CIFAR10

    CIFAR_Test = torchvision.datasets.CIFAR10("/Users/adithyagiri/Desktop/STS/Repo/Datasets/CIFAR-10", train = False, download=True)
    return (CIFAR_Train, CIFAR_Test, CIFAR_Val)

"""Pytorch Dataset class that allows for easy and efficient dataloading of the CIFAR-10 dataset. Has the ability to downsample images for training without patching."""
class CIFAR_Conv(Dataset):
    def __init__(self, dataset, train = True):
        self.CIFAR = dataset
        if(train):
          self.tensor_transform = transforms.Compose([
              transforms.Resize((32,32)),
              transforms.ToTensor(),
              transforms.Normalize(0,1)
          ])
        else:
          self.tensor_transform = transforms.Compose([
              transforms.Resize((32,32)),
              transforms.ToTensor(),
              transforms.Normalize(0,1)
          ])
    def __getitem__(self, index):
        image = self.CIFAR.__getitem__(index)
        return (self.tensor_transform(image[0]),image[1])
    def __len__(self):
        return len(self.CIFAR)
    
"""Initialize all the datasets and dataloaders"""  
def return_datasets(size, batch_size):
    CIFAR_Train, CIFAR_Test, CIFAR_Val = download_data(size=size)
    CIFAR_Train = CIFAR_Conv(dataset = CIFAR_Train, train = True)
    CIFAR_Val = CIFAR_Conv(dataset = CIFAR_Val, train = False)
    CIFAR_Test = CIFAR_Conv(dataset = CIFAR_Test, train = False)
    CIFAR_Convloader = DataLoader(CIFAR_Train, batch_size = batch_size, shuffle = True)
    CIFAR_Valconvloader = DataLoader(CIFAR_Val, batch_size = batch_size, shuffle = True)
    CIFAR_Testconvloader = DataLoader(CIFAR_Test, batch_size = batch_size, shuffle = True)
    return (CIFAR_Convloader, CIFAR_Valconvloader, CIFAR_Testconvloader)