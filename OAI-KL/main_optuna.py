import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import optuna

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR

from dataset import ImageDataset
from early_stop import EarlyStopping
from model import model_return

def train_for_kfold(model, dataloader, criterion, optimizer, scheduler, fold, epoch):
    train_loss = 0.0
    model.train() # Model을 Train Mode로 변환 >> Dropout Layer 같은 경우 Train시 동작 해야 함
    with torch.set_grad_enabled(True): # with문 : 자원의 효율적 사용, 객체의 life cycle을 설계 가능, 항상(True) gradient 연산 기록을 추적
        for batch in tqdm(dataloader, desc=f'Fold {fold} Epoch {epoch} Train', unit='Batch'):
            optimizer.zero_grad() # 반복 시 gradient(기울기)를 0으로 초기화, gradient는 += 되기 때문
            image, labels = batch['image'].cuda(), batch['target'].cuda() # Tensor를 GPU에 할당
            
            # labels = F.one_hot(labels, num_classes=5).float() # nn.MSELoss() 사용 시 필요
            output = model(image) # image(data)를 model에 넣어서 hypothesis(가설) 값을 획득
            
            loss = criterion(output, labels) # Error, Prediction Loss 계산
            train_loss += loss.item() # loss.item()을 통해 Loss의 스칼라 값을 가져온다.

            loss.backward() # Prediction Loss를 Back Propagation으로 계산
            optimizer.step() # optimizer를 이용해 Loss를 효율적으로 최소화 할 수 있게 Parameter 수정
            
        scheduler.step()
            
    return train_loss

def val_for_kfold(model, dataloader, criterion, fold, epoch):
    val_loss = 0.0
    model.eval() # Model을 Eval Mode로 전환 >> Dropout Layer 같은 경우 Eval시 동작 하지 않아야 함
    with torch.no_grad(): # gradient 연산 기록 추적 off
        for batch in tqdm(dataloader, desc=f'Fold {fold} Epoch {epoch} Valid', unit='Batch'):
            image, labels = batch['image'].cuda(), batch['target'].cuda()
            
            # labels = F.one_hot(labels, num_classes=5).float() # nn.MSELoss() 사용 시 필요
            output = model(image)
            
            loss = criterion(output, labels)
            val_loss += loss.item()
             
    return val_loss

def train(train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf):
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)), labels), start=1):
        # Data Load에 사용되는 index, key의 순서를 지정하는데 사용, Sequential , Random, SubsetRandom, Batch 등 + Sampler
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)        
        # Data Load
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
        
        model_ft = model_return(args)
                
        if torch.cuda.device_count() > 1:
            model_ft = nn.DataParallel(model_ft) # model이 여러 대의 gpu에 할당되도록 병렬 처리
        model_ft.cuda() # Model을 GPU에 할당

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Loss Function
        # criterion = nn.MSELoss()
        # criterion = my_ce_mse_loss
        
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model_ft.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model_ft.parameters(), lr=args.learning_rate)
            
        scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
        
        history = {'train_loss': [], 'val_loss': []}
            
        patience = 15
        delta = 0.15
        early_stopping = EarlyStopping(args, patience=patience, verbose=True, delta=delta)
        
        for epoch in range(1, epochs + 1):
            train_loss = train_for_kfold(model_ft, train_loader, criterion, optimizer, scheduler, fold, epoch)
            val_loss = val_for_kfold(model_ft, val_loader, criterion, fold, epoch)
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            print(f"Epoch: {epoch}/{epochs} \t Avg Train Loss: {train_loss:.3f} \t Avg Valid Loss: {val_loss:.3f}")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            early_stopping(val_loss, model_ft, args, fold, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        foldperf[f"fold{fold}"] = history
    
    tl_f, vall_f = [], []

    for f in range(1, k+1):
        tl_f.append(np.mean(foldperf[f'fold{f}']['train_loss']))
        vall_f.append(np.mean(foldperf[f'fold{f}']['val_loss']))

    print()
    print(f"Performance of {k} Fold Cross Validation")
    print(f"Avg Train Loss: {np.mean(tl_f):.3f} \t Avg Valid Loss: {np.mean(vall_f):.3f}")
    
    return vall_f

def objective(trial):
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD']) # optimizer 설정
    if optimizer == 'Adam':
        lr = trial.suggest_float('lr', 1e-4, 1e-3)  # learning rate 설정
    if optimizer == 'SGD':
        lr = trial.suggest_float('lr', 1e-3, 1e-2)  # learning rate 설정
    batch_size = int(trial.suggest_categorical('batch_size', [16, 32])) # batch size 설정
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', dest='model_type', action='store')
    parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action='store')
    args = parser.parse_args()
    
    args.optimizer = optimizer
    args.learning_rate = lr
    
    image_size_tuple = (args.image_size, args.image_size)
    
    print(f"Model Type : {args.model_type}")
    print(f"Image Size : {image_size_tuple}")
    print(f"Optimizer : {optimizer}")
    print(f"Learning Rate : {lr}")
    print(f"Batch Size : {batch_size}")
    
    train_csv = pd.read_csv('./KneeXray/train/train.csv')

    train_transform = A.Compose([
                    A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20, p=1),
                    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # -1 ~ 1의 범위를 가지도록 정규화
                    ToTensorV2() # 0 ~ 1의 범위를 가지도록 정규화
                    ])
    val_transform = A.Compose([
                    A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_CUBIC, p=1),
                    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # -1 ~ 1의 범위를 가지도록 정규화
                    ToTensorV2() # 0 ~ 1의 범위를 가지도록 정규화
                    ])
    train_dataset = ImageDataset(train_csv, transforms=train_transform)
    val_dataset = ImageDataset(train_csv, transforms=val_transform)
    
    epochs = 1
    k = 2
    torch.manual_seed(42)
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = train_dataset.get_labels()
    foldperf = {}

    val_l = train(train_dataset, val_dataset, args, batch_size, epochs, k, splits, labels, foldperf)
    
    return np.mean(val_l)

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)