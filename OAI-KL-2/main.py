import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import transforms
from torch import nn, optim
from tqdm.notebook import tqdm
from torchvision import models
import torch.nn.functional as F
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
import os

train = pd.read_csv('./KneeXray/Train.csv')
#train['data'] = './KneeXray/train/' + train['data']

test = pd.read_csv('./KneeXray/Test.csv')
#test['data'] = './KneeXray/test/' + test['data']

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

transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                               ])

train_data = ImageDataset(train, transforms = transform)
#test_data = ImageDataset(test)

#trainloader = DataLoader(train_data, batch_size = 32, shuffle=True)
#testloader = DataLoader(test_data, batch_size = 32, shuffle=False)


'''
for param in model_ft.parameters():
    param.requires_grad = False


num_ftrs = model_ft.fc.in_features
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.fc = nn.Linear(num_ftrs, 5)
'''


dataset = ConcatDataset([train_data])

epochs = 100
batch_size = 16
k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

def train_for_kfold(model, dataloader, criterion, optimizer):
    with torch.set_grad_enabled(True):
        train_loss = 0.0
        for batch in (dataloader):
            optimizer.zero_grad()
            image, labels = batch['image'].cuda(), batch['target'].cuda()

            output = model_ft(image)
            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            
    return train_loss

def test_for_kfold(model, dataloader, criterion):
    test_loss = 0.0
    model_ft.eval()
    for batch in (dataloader):
        with torch.no_grad():
             image, labels = batch['image'].cuda(), batch['target'].cuda()
             output = model_ft(image)
             loss = criterion(output, labels)
             test_loss += loss.item()
             
    return test_loss

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()

    optimizer = optim.Adam(model_ft.parameters())
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'test_loss': []}
    early_stopping = 10
    count = 0
    val_min_loss = 100000000000.0
    
    for epoch in range(epochs):
        train_loss = train_for_kfold(model_ft, train_loader, criterion,optimizer)
        test_loss = test_for_kfold(model_ft, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        #train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        #test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(epoch + 1, epochs, train_loss, test_loss))
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if (epoch + 1) % 5 == 0 and val_min_loss > test_loss:
            print('model saving.....')
            val_min_loss = test_loss 
            torch.save(model_ft,'./models/kfold_CNN_{}fold_{}epoch.pt'.format(fold, epoch + 1))
        else:
            count = count + 1
            
        '''
        if count == early_stopping:
            break
        '''
        
    foldperf['fold{}'.format(fold+1)] = history  
    
testl_f,tl_f =[],[]
k=5

for f in range(1,k+1):

     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f}".format(np.mean(tl_f),np.mean(testl_f)))