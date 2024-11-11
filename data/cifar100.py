import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
class CIFAR100Dataset(Dataset):
    """
    CIFAR100Dataset is a custom dataset class for handling the CIFAR-100 dataset.

    This class loads the CIFAR-100 dataset using torchvision.

    Attributes:
        transform (callable): A function/transform to apply to the data.

    Methods:
        __init__(train, transform):
            Initializes the CIFAR100Dataset with the specified parameters.
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Returns the data and target at the specified index.
    """
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.cifar100 = datasets.CIFAR100(root='./data', train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        return self.cifar100[idx]