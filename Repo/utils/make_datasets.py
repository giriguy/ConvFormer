import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import sys
import math
import random
import timm
from timm.data import create_transform
"""Loads the CIFAR-10 Dataset, and splits it into two sections. This allows for reducing dataset size to train models on smaller dataset sizes and analyze model scalability."""
def download_data_cifar(size):
    CIFAR10Orig = torchvision.datasets.CIFAR10("/workspace/ConvFormer/Repo/Datasets/CIFAR-10", train=True, download=True)
    seed = 1
    data_len = 50000
    val_size = 10000
    CIFAR10, CIFAR_Val, Remaining = torch.utils.data.random_split(
            CIFAR10Orig, [size, val_size, data_len-val_size-size], generator=torch.Generator().manual_seed(seed))

    print(len(CIFAR10))
    CIFAR_Train = CIFAR10

    CIFAR_Test = torchvision.datasets.CIFAR10("/workspace/ConvFormer/Repo/Datasets/CIFAR-10", train = False, download=True)
    return (CIFAR_Train, CIFAR_Test, CIFAR_Val)
"""Loads the imagenet-100 Dataset, and splits it into two sections. This allows for reducing dataset size to train models on smaller dataset sizes and analyze model scalability."""
def download_data_imagenet(size=None):
    image_net_set = torchvision.datasets.ImageNet(root = '/workspace/ConvFormer/Repo/Datasets/', split = 'train')
    seed = 1
    data_len = len(image_net_set)
    print(data_len)
    if(size == None):
        size = data_len
    IN_Train, Remaining = torch.utils.data.random_split(
            image_net_set, [size, data_len-size], generator=torch.Generator().manual_seed(seed))
    IN_Val = torchvision.datasets.ImageNet(root = '/workspace/ConvFormer/Repo/Datasets/', split = 'val')
    return (IN_Train, IN_Val, IN_Val)

"""Pytorch Dataset class that allows for easy and efficient dataloading of the CIFAR-10 dataset. Has the ability to downsample images for training without patching."""
class CIFAR_Conv(Dataset):
    def __init__(self, dataset, train = True):
        self.CIFAR = dataset
        if(train):
          self.tensor_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.CenterCrop(16),
            transforms.RandomCrop(16, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
          ])
        else:
          self.tensor_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.CenterCrop(16),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
          ])
    def __getitem__(self, index):
        image, label = self.CIFAR.__getitem__(index)
        return (self.tensor_transform(image),label)
    def __len__(self):
        return len(self.CIFAR)
    
"""Pytorch Dataset class that allows for easy and efficient dataloading of the IN dataset. Has the ability to downsample images for training without patching."""
class IN_Conv(Dataset):
    def __init__(self, dataset, train = True):
        self.IN = dataset
        if(train):
          self.tensor_transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1
          )
          # self.tensor_transform = transforms.Compose([
          #   transforms.RandomResizedCrop(size=224),
          #   transforms.RandomHorizontalFlip(),
          #   transforms.RandAugment(num_ops = 2, magnitude = 9),
          #   transforms.ToTensor(),
          #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          #   transforms.RandomErasing(p=0.25),
          #   ])
        else:
          # self.tensor_transform = create_transform(
          #   input_size=224,
          #   is_training=False,
          #   color_jitter=0.3,
          #   auto_augment='rand-m9-mstd0.5-inc1',
          #   interpolation='bicubic',
          #   re_prob=0.25,
          #   re_mode='pixel',
          #   re_count=1
          # )
          self.tensor_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])
    def __getitem__(self, index):
        example = self.IN.__getitem__(index)
        image = example[0].convert('RGB')
        label = example[1]
        return (self.tensor_transform(image),label)
    def __len__(self):
        return len(self.IN)
    
"""Initialize all the datasets and dataloaders"""  
def return_datasets_cifar(size, batch_size):
    CIFAR_Train, CIFAR_Test, CIFAR_Val = download_data_cifar(size=size)
    CIFAR_Train = CIFAR_Conv(dataset = CIFAR_Train, train = True)
    CIFAR_Val = CIFAR_Conv(dataset = CIFAR_Val, train = False)
    CIFAR_Test = CIFAR_Conv(dataset = CIFAR_Test, train = False)
    CIFAR_Convloader = DataLoader(CIFAR_Train, batch_size = batch_size, shuffle = True, num_workers=4)
    CIFAR_Valconvloader = DataLoader(CIFAR_Val, batch_size = batch_size, shuffle = True)
    CIFAR_Testconvloader = DataLoader(CIFAR_Test, batch_size = batch_size, shuffle = True)
    return (CIFAR_Convloader, CIFAR_Valconvloader, CIFAR_Testconvloader)

def return_datasets_IN(size=None, batch_size=128):
    IN_Train, IN_Test, IN_Val = download_data_imagenet(size=size)
    IN_Train = IN_Conv(dataset = IN_Train, train = True)
    IN_Val = IN_Conv(dataset = IN_Val, train = False)
    IN_Test = IN_Conv(dataset = IN_Test, train = False)
    IN_Convloader = DataLoader(IN_Train, batch_size = batch_size, shuffle = True, num_workers=64)
    IN_Valconvloader = DataLoader(IN_Val, batch_size = batch_size, shuffle = True, num_workers=64)
    IN_Testconvloader = DataLoader(IN_Test, batch_size = batch_size, shuffle = True)
    return (IN_Convloader, IN_Valconvloader, IN_Testconvloader)