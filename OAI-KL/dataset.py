import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, df, transforms = transforms.Compose([transforms.ToTensor()])):
        self.path = df['data']
        self.transforms = transforms
        if 'label' in df:
            self.target = df['label']
        else:
            self.target = None

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(380, 380), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image)

        if self.target is not None:
            return {
                    'image': image.float(),
                    'target': torch.tensor(self.target[idx], dtype = torch.long)
                   }
        else: 
            return {
                    'image': image.float()
                   }