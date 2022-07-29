import cv2
import torch
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
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD + _COLOR, _GRAYSCALE, _UNCHANGED
        image = cv2.resize(image, dsize=(380, 380), interpolation=cv2.INTER_CUBIC) # interpolation : 보간법
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # opencv에서는 BGR 방식으로 표현 cv2.COLOR + _BGR2RGB, BGR2GRAY 등
        
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