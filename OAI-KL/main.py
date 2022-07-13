import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from dataset import ImageDataset
from early_stop import EarlyStopping

def train_for_kfold(model, dataloader, criterion, optimizer):
    train_loss = 0.0
    model.train() # model을 train mode로 변환 >> Dropout Layer 같은 경우 train시 동작 해야 함
    with torch.set_grad_enabled(True): # with문 : 자원의 효율적 사용, 객체의 life cycle을 설계 가능, 항상(True) gradient 연산 기록을 추적
        for batch in (dataloader):
            optimizer.zero_grad() # 반복 시 gradient(기울기)를 0으로 초기화, gradient는 += 되기 때문
            image, labels = batch['image'].cuda(), batch['target'].cuda() # tensor를 gpu에 할당

            output = model(image) # image(data)를 model에 넣어서 hypothesis(가설) 값을 획득
            loss = criterion(output, labels) # error, prediction loss를 계산
            train_loss += loss.item() # loss.item()을 통해 loss의 스칼라 값을 가져온다.

            loss.backward() # prediction loss를 backpropagation으로 계산
            optimizer.step() # optimizer를 이용해 loss를 효율적으로 최소화 할 수 있게 parameter 수정
            
    return train_loss

def test_for_kfold(model, dataloader, criterion):
    test_loss = 0.0
    model.eval() # model을 eval mode로 전환 >> Dropout Layer 같은 경우 eval시 동작 하지 않아야 함
    with torch.no_grad(): # gradient 연산 기록 추적 off 
        for batch in (dataloader):
             image, labels = batch['image'].cuda(), batch['target'].cuda()
             
             output = model(image)
             loss = criterion(output, labels)
             test_loss += loss.item()
             
    return test_loss

def train(dataset, epochs, batch_size, k, splits, foldperf):

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        patience = 30
        delta = 0.2
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)
    
        train_sampler = SubsetRandomSampler(train_idx) # data load에 사용되는 index, key의 순서를 지정하는데 사용, Sequential , Random, SubsetRandom, Batch 등 + Sampler
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler) # Data Load
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        # model_ft = EfficientNet.from_name('efficientnet-b5', num_classes=5)
        model_ft = EfficientNet.from_pretrained('efficientnet-b5', num_classes=5)
        model_ft = nn.DataParallel(model_ft) # model이 여러 대의 gpu에 할당되도록 병렬 처리
        model_ft = model_ft.cuda() # model을 gpu에 할당

        optimizer = optim.Adam(model_ft.parameters(), lr=0.0006) # optimizer
        criterion = nn.CrossEntropyLoss() # loss function
        history = {'train_loss': [], 'test_loss': []}
        
        print('Fold {}'.format(fold + 1))
    
        for epoch in range(epochs):
            train_loss = train_for_kfold(model_ft, train_loader, criterion, optimizer)
            test_loss = test_for_kfold(model_ft, test_loader, criterion)

            train_loss = train_loss / len(train_loader)
            # train_acc = train_correct / len(train_loader.sampler)*100
            test_loss = test_loss / len(test_loader)
            # test_acc = test_correct / len(test_loader.sampler)*100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(epoch + 1, epochs, train_loss, test_loss))
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            
            early_stopping(test_loss, model_ft, fold, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        foldperf['fold{}'.format(fold+1)] = history  
    
    testl_f,tl_f =[],[]
    k=1

    for f in range(1, k+1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f}".format(np.mean(tl_f),np.mean(testl_f)))

if __name__ == '__main__':
    train_data = pd.read_csv('./KneeXray/Train.csv')
    transform = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p = 0.5),
                                    transforms.RandomRotation(10),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                  ])
    dataset = ImageDataset(train_data, transforms=transform)
    batch_size = 1
    epochs = 100
    k = 2
    torch.manual_seed(42)
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    foldperf = {}

    train(dataset, epochs, batch_size, k, splits, foldperf)