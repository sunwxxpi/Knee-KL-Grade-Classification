import pandas as pd
import os
from sklearn.metrics import accuracy_score

csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'])
Test_correct_label = csv['label']
Test_correct_label_list = Test_correct_label.values.tolist()

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]

for i in submission_list_csv:
    csv = pd.read_csv('{}{}'.format(submission_path, i), names=['data', 'label'])
    submission_label = csv['label']
    submission_label_list = submission_label.values.tolist()
    
    print('{}'.format(i))
    score = accuracy_score(Test_correct_label_list, submission_label_list)
    print('accuracy_score : {}'.format(score))
    print()