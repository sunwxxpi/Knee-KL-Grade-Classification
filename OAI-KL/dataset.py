import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, df, transforms=transforms.Compose([transforms.ToTensor()])):
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
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transforms:
            image = self.transforms(image)

        if self.target is not None:
            return {
                    'image': image.float(),
                    'target': torch.tensor(self.target[idx], dtype=torch.long)
                    }
        else: 
            return {
                    'image': image.float()
                    }