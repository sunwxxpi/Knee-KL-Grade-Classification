import pandas as pd
import os
import torch
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset

test_csv = pd.read_csv('./KneeXray/Test_correct.csv') # _cn _clahe 등, 수정 O, 수정 x -> 결과 비교

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                ])
test_data = ImageDataset(test_csv, transforms=transform)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

transform_tta = tta.Compose([
                            tta.HorizontalFlip()
                            ])

model_path = './models/'
submission_path = './submission/'
model_list = os.listdir(model_path)
model_list_pt = [file for file in model_list if file.endswith(".pt")]

for i in model_list_pt: 
    preds = []
    probs_correct = []
    probs_predict = []
    probs_0, probs_1, probs_2, probs_3, probs_4 = [], [], [], [], []
    
    model_ft = torch.load('{}{}'.format(model_path, i))
    model_ft = tta.ClassificationTTAWrapper(model_ft, transform_tta)

    for batch in testloader:
        with torch.no_grad():
            image = batch['image'].cuda()
            target = batch['target'].cuda()
            output = model_ft(image)
            preds.extend([i.item() for i in torch.argmax(output, axis=1)]) # tensor 자료형의 예측 라벨 값을 list로 뽑아 preds = []에 extend
            
            softmax = nn.Softmax(dim=1)
            softmax_output = softmax(output).detach().cpu().numpy()
            probs_correct.append(softmax_output[0][int(target)]) # softmax 출력값 중 correct label에 해당하는 확률값을 probs_correct = []에 append
            probs_predict.append(max(softmax_output[0])) # softmax 출력값 중 predict label에 해당하는 확률값을 probs_predict = []에 append
            for k in range(5):
                globals()['probs_{}'.format(k)].append(softmax_output[0][k]) # softmax 출력값 중 class에 해당하는 확률값을 probs_{} = []에 append
        
    submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test_csv['data']], 'label':preds, 'prob_correct':probs_correct, 'prob_predict':probs_predict, 'prob_0':probs_0, 'prob_1':probs_1, 'prob_2':probs_2, 'prob_3':probs_3, 'prob_4':probs_4})

    fold_and_epoch = i[10:-3]
    submit.to_csv('{}{}_submission.csv'.format(submission_path, fold_and_epoch), index=False)
    print('save {}{}_submission.csv'.format(submission_path, fold_and_epoch))