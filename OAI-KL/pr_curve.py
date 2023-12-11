import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

test_csv = pd.read_csv('./KneeXray/test/test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

test_image_num = len(test_correct_labels_list)

submission_list = os.listdir('./submission')
submission_list_csv = [file for file in submission_list if file.endswith('.csv')]

for haha in submission_list_csv:
    submission_csv = pd.read_csv(f'./submission/{haha}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    submission_labels = submission_csv['label']
    submission_labels_list = submission_labels.values.tolist()

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
        globals()[f'image_{i}'] = [0 for _ in range(5)]
            
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
        display = PrecisionRecallDisplay.from_predictions(correct_label[:, i], probs[:, i], name=f'Class {i}', ax=axs)
    plt.title(f"Optimized {haha.split('_')[1].split('.')[0]}", fontsize=19)
    plt.xlabel("Recall", fontsize=17.5)
    plt.ylabel("Precision", fontsize=17.5)

    plt.show()