import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, classification_report

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

test_image_num = len(test_correct_labels_list)

fold = 10
epoch = 12

submission_csv = pd.read_csv(f'./submission/{fold}fold_epoch{epoch}_submission.csv', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
submission_labels = submission_csv['label']
submission_labels_list = submission_labels.values.tolist()

normalize = 'true'
# normalize = None
score = accuracy_score(test_correct_labels_list, submission_labels_list)
matrix = confusion_matrix(test_correct_labels_list, submission_labels_list, normalize=normalize)

plt.figure(1, figsize=(7, 7.5))
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.title('3 Model\nBase Ensemble Network', size=19)
plt.xlabel(f"Predicted label\n\nAccuracy = {score * 100: .2f}%", size=15)
plt.ylabel("True label", size=15)
plt.xticks(range(5), (0, 1, 2, 3, 4), size=11)
plt.yticks(range(5), (0, 1, 2, 3, 4), size=11)
plt.colorbar(fraction=0.05, pad=0.05, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])

fmt = '.3f' if normalize=='true' else 'd'
threshold = 0.5 if normalize=='true' else 450
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, format(matrix[i, j], fmt), ha='center', va='center', color='white' if matrix[i, j] > threshold else 'black', size=13)  # Horizontal Alignment

correct_label = [[0 for _ in range(5)] for _ in range(test_image_num)]

for i in range(test_image_num):
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

for i in range(test_image_num):
        globals()[f'image_{i}'] = [0 for j in range(5)]
        
for i in range(5):
        submission_probs = submission_csv[f'prob_{i}']
        submission_probs_list = submission_probs.values.tolist()
        
        for j in range(test_image_num):
            globals()[f'image_{j}'][i] = submission_probs_list[j]
            
for i in range(test_image_num):
        probs.append(globals()[f'image_{i}'])

correct_label = np.array(correct_label)
probs = np.array(probs)

fig, axs = plt.subplots(figsize=(6, 6))
for i in range(5):
    display = RocCurveDisplay.from_predictions(correct_label[:, i], probs[:, i], name=f'Class {i}', ax=axs)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

fig, axs = plt.subplots(figsize=(6, 6))
for i in range(5):
    display = PrecisionRecallDisplay.from_predictions(correct_label[:, i], probs[:, i], name=f'Class {i}', ax=axs)
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.show()

report = classification_report(test_correct_labels_list, submission_labels_list, digits=4)
print(report)