import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

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
# roc = roc_curve(Test_correct_label_list, , )

print(report)
                         
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