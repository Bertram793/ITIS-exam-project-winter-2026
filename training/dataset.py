import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class FruitDatabase(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])