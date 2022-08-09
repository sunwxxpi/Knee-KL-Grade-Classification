import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, classification_report

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

fold = 10
epoch = 10

submission_csv = pd.read_csv('./submission/{}fold_epoch{}_submission.csv'.format(fold, epoch), names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
submission_labels = submission_csv['label']
submission_labels_list = submission_labels.values.tolist()

normalize = 'true'
score = accuracy_score(test_correct_labels_list, submission_labels_list)
matrix = confusion_matrix(test_correct_labels_list, submission_labels_list, normalize=normalize)

title = 'Confusion Matrix'
plt.figure(1, figsize=(7, 7.5))
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
        plt.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i, j] > thresh else "black")  # Horizontal Alignment

correct_label = [[0 for j in range(5)] for i in range(1656)]

for i in range(1656):
    if test_correct_labels_list[i] == 0:
        correct_label[i][0] = 1
    elif test_correct_labels_list[i] == 1:
        correct_label[i][1] = 1
    elif test_correct_labels_list[i] == 2:
        correct_label[i][2] = 1
    elif test_correct_labels_list[i] == 3:
        correct_label[i][3] = 1
    elif test_correct_labels_list[i] == 4:
        correct_label[i][4] = 1
        
probs = []

for i in range(1656):
        globals()['image_{}'.format(i)] = [0 for j in range(5)]
        
for i in range(5):
        submission_probs = submission_csv['prob_{}'.format(i)]
        submission_probs_list = submission_probs.values.tolist()
        
        for j in range(1656):
            globals()['image_{}'.format(j)][i] = submission_probs_list[j]
            
for i in range(1656):
        probs.append(globals()['image_{}'.format(i)])

correct_label = torch.tensor(correct_label)  
probs = torch.tensor(probs)

fig, axs = plt.subplots(figsize=(6, 6))
for i in range(5):
    display = RocCurveDisplay.from_predictions(correct_label[:, i], probs[:, i], name='Class {}'.format(i), ax=axs)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

report = classification_report(test_correct_labels_list, submission_labels_list, digits=4)
print(report)