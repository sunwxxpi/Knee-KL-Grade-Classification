import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, df, transforms=A.Compose([ToTensorV2()])):
        self.data = df['data']
        self.label = df['label']
        self.transforms = transforms
        
        if 'label' in df:
            self.target = df['label']
        else:
            self.target = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # OpenCV에서는 BGR 방식으로 표현

        if self.transforms:
            image = self.transforms(image=image)['image']

        if self.target is not None:
            return {
                'image': image.float(),
                'target': torch.tensor(self.target[idx], dtype=torch.long)
                }
        else: 
            return {
                'image': image.float()
                }
            
    def get_labels(self):
        return self.label.values