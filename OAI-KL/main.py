import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler
from torchvision import transforms
from torch import nn, optim
from torchvision import models
import torch.nn.functional as F
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
import random
from dataset import ImageDataset
from early_stop import EarlyStopping

def train_for_kfold(model, dataloader, criterion, optimizer):
    model.train()
    with torch.set_grad_enabled(True):
        train_loss = 0.0
        for batch in (dataloader):
            optimizer.zero_grad()
            image, labels = batch['image'].cuda(), batch['target'].cuda()

            output = model(image)
            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            
    return train_loss

def test_for_kfold(model, dataloader, criterion):
    test_loss = 0.0
    model.eval()
    for batch in (dataloader):
        with torch.no_grad():
             image, labels = batch['image'].cuda(), batch['target'].cuda()
             output = model(image)
             loss = criterion(output, labels)
             test_loss += loss.item()
             
    return test_loss

def train(dataset, epochs, batch_size, k, splits, foldperf):

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        patience = 20
        early_stopping = EarlyStopping(patience = patience, verbose = True)
    
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
        test_loader = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler)
    
        model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)
        model_ft = nn.DataParallel(model_ft)
        model_ft = model_ft.cuda()

        optimizer = optim.Adam(model_ft.parameters())
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'test_loss': []}
        
        print('Fold {}'.format(fold + 1))
    
        for epoch in range(epochs):
            train_loss = train_for_kfold(model_ft, train_loader, criterion, optimizer)
            test_loss = test_for_kfold(model_ft, test_loader, criterion)

            train_loss = train_loss / len(train_loader)
            #train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader)
            #test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(epoch + 1, epochs, train_loss, test_loss))
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            
            early_stopping(test_loss, model_ft, fold, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        foldperf['fold{}'.format(fold+1)] = history  
    
    testl_f,tl_f =[],[]
    k=5

    for f in range(1, k+1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f}".format(np.mean(tl_f),np.mean(testl_f)))

if __name__ == '__main__':
    train_data = pd.read_csv('./KneeXray/Train.csv')
    transform = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                  ])
    
    dataset = ImageDataset(train_data, transforms = transform)
    torch.manual_seed(42)
    epochs = 100
    batch_size = 32
    k = 5
    splits = KFold(n_splits = k, shuffle = True, random_state = 42)
    foldperf = {}

    train(dataset, epochs, batch_size, k, splits, foldperf)