from numpy import isin
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset

test = pd.read_csv('./KneeXray/Test_correct.csv') # _cn _clahe 등, 수정 O, 수정 x -> 결과 비교

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                               ])

test_data = ImageDataset(test, transforms=transform)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

model_path = './models/'
submission_path = './submission/'
model_list = os.listdir(model_path)
model_list_pt = [file for file in model_list if file.endswith(".pt")]

for i in model_list_pt: 
    preds = []
    model_ft = torch.load('{}{}'.format(model_path, i))

    for batch in testloader:
        with torch.no_grad():
            image = batch['image'].cuda()
            output = model_ft(image)
            preds.extend([i.item() for i in torch.argmax(output, axis=1)]) # tensor 자료형의 예측 라벨 값을 list로 뽑아 preds = []에 extend
    
    submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test['data']], 'label':preds})

    fold_and_epoch = i[10:-3]
    submit.to_csv('{}{}_submission.csv'.format(submission_path, fold_and_epoch), index=False)
    print('save {}{}_submission.csv'.format(submission_path, fold_and_epoch))