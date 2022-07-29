import pandas as pd
import os
from sklearn.metrics import classification_report

csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'])
Test_correct_label = csv['label']
Test_correct_label_list = Test_correct_label.values.tolist()

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]

label = ['0', '1', '2', '3', '4'] # 라벨 설정

for i in submission_list_csv:
    csv = pd.read_csv('{}{}'.format(submission_path, i), names=['data', 'label'])
    submission_label = csv['label']
    submission_label_list = submission_label.values.tolist()
    
    print('{}'.format(i))
    report = classification_report(Test_correct_label_list, submission_label_list, labels=label, digits=4) # micro avg f1 score = accuracy score
    print(report)
    print()