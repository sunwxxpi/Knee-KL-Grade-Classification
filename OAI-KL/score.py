import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

fold = 5
epoch = 4

submission_csv = pd.read_csv('./submission/{}fold_epoch{}_submission.csv'.format(fold, epoch), names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)

submission_labels = submission_csv['label']
submission_labels_list = submission_labels.values.tolist()
# submission_probs = csv['prob']
# submission_probs_list = submission_probs.values.tolist()

normalize = 'true'
score = accuracy_score(test_correct_labels_list, submission_labels_list)
report = classification_report(test_correct_labels_list, submission_labels_list, digits=4) # micro avg f1 score = accuracy score
matrix = confusion_matrix(test_correct_labels_list, submission_labels_list, normalize=normalize)
print(report)

# for i in range(5):
#     fprs, tprs, _ = roc_curve(test_correct_labels_list[:,i], submission_prob_list[:,i]) #calculate fprs and tprs for each class
#     plt.plot(fprs,tprs,label='{}'.format(i)) #plot roc curve of each class
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
       
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
        plt.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i, j] > thresh else "black")  # Horizontal Alignment

plt.show()