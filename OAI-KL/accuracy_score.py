import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader

csv = pd.read_csv('./KneeXray/Test_correct.csv', names = ['data', 'label'])
Test_correct_label = csv['label']
Test_correct_label_list = Test_correct_label.values.tolist()

fold = 5
epoch = 38

csv = pd.read_csv('./submission/{}fold_epoch{}_submission.csv'.format(fold, epoch), names=['data', 'label'])
submission_label = csv['label']
submission_label_list = submission_label.values.tolist()

label = ['0', '1', '2', '3', '4'] # 라벨 설정
normalize = 'true'
score = accuracy_score(Test_correct_label_list, submission_label_list)
report = classification_report(Test_correct_label_list, submission_label_list, labels=label, digits=4) # micro avg f1 score = accuracy score
matrix = confusion_matrix(Test_correct_label_list, submission_label_list, labels=label, normalize=normalize)
'''
test = pd.read_csv('./KneeXray/Test_correct.csv') # _cn _clahe 등, 수정 O, 수정 x -> 결과 비교

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                               ])

test_data = ImageDataset(test, transforms=transform)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

model_path = './models/'
model_list = os.listdir(model_path)
model_list_pt = [file for file in model_list if file.endswith(".pt")]

for i in model_list_pt: 
    confidence = []
    model_ft = torch.load('{}{}'.format(model_path, i))

    for batch in testloader:
        with torch.no_grad():
            image = batch['image'].cuda()
            output = model_ft(image)
            softmax = nn.Softmax(dim=1)
            softmax_output = np.max(softmax(output).detach().cpu().numpy())
            confidence.append(softmax_output)
            
auroc = roc_auc_score(Test_correct_label_list, confidence, average='macro', multi_class='ovr')
'''
print(report)
print(matrix)
# print(auroc)
                         
title = 'Confusion Matrix'
plt.figure(figsize=(7, 7.5))
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title, size=12)
plt.colorbar(fraction=0.05, pad=0.05)
tick_marks = np.arange(5, 5)
plt.xlabel('Predicted label\n\naccuracy={:0.5f}\nTotal : 1656\n0 : 639          1 : 296          2 : 447          3 : 223          4 : 51'.format(score))
plt.ylabel('True label')
plt.xticks(np.arange(5), ('0', '1', '2', '3', '4'))
plt.yticks(np.arange(5), ('0', '1', '2', '3', '4'))

fmt = '.3f' if normalize else 'd'
thresh = 200
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i, j] > thresh else "black")  # horizontal alignment

plt.show()