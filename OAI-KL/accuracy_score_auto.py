import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix

csv = pd.read_csv('./KneeXray/Test_correct.csv', names = ['data', 'label'])
Test_correct_label = csv['label']
Test_correct_label_list = Test_correct_label.values.tolist()

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]

for i in submission_list_csv:
    csv = pd.read_csv('{}{}'.format(submission_path, i), names = ['data', 'label'])
    submission_label = csv['label']
    submission_label_list = submission_label.values.tolist()
    
    '''
    print('{}'.format(i))
    print()
    
    score = accuracy_score(Test_correct_label_list, submission_label_list)
    print('accuracy_score : {}'.format(score))
    label = ['0', '1', '2', '3', '4'] # 라벨 설정
    normalize = 'true'
    matrix = confusion_matrix(Test_correct_label_list, submission_label_list, labels=label, normalize=normalize)
    print(matrix)
    
    print()
    print()
    '''
    
    print('{}'.format(i))
    score = accuracy_score(Test_correct_label_list, submission_label_list)
    print('accuracy_score : {}'.format(score))
    print()