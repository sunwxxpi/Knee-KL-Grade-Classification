import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from dataset import ImageDataset

def trainer(model, dataloader, criterion, optimizer):
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

def train(dataset, epochs, batch_size):
    train_sampler = RandomSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler) # Data Load
    
    model_ft = EfficientNet.from_pretrained('efficientnet-b5', num_classes=5)
    model_ft = nn.DataParallel(model_ft) # model이 여러 대의 gpu에 할당되도록 병렬 처리
    model_ft = model_ft.cuda() # model을 gpu에 할당

    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model_ft.parameters(), lr=0.001) # optimizer

    for epoch in range(epochs):
        train_loss = trainer(model_ft, train_loader, criterion, optimizer)
        train_loss = train_loss / len(train_loader)
        # train_loss = train_correct / len(train_loader.sampler)*100
        
        torch.save(model_ft,'./models/kfold_CNN_{}fold_epoch{}.pt'.format(1, epoch + 1))
        print("Epoch:{}/{} AVG Training Loss:{:.3f}".format(epoch + 1, epochs, train_loss))

if __name__ == '__main__':
    train_csv = pd.read_csv('./KneeXray/Train.csv') # _cn _clahe 등, 수정 필요
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(20),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                    ])
    dataset = ImageDataset(train_csv, transforms=transform)
    batch_size = 32
    epochs = 100

    train(dataset, epochs, batch_size)