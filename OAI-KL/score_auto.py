import os
import natsort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]
submission_list_csv = natsort.natsorted(submission_list_csv)

sum_accuracy = 0
sum_f1_macro = 0
sum_f1_weighted = 0

for i in submission_list_csv:
    submission_csv = pd.read_csv('{}{}'.format(submission_path, i), names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    submission_labels = submission_csv['label']
    submission_labels_list = submission_labels.values.tolist()
    
    print('{}'.format(i))
    accuracy = accuracy_score(test_correct_labels_list, submission_labels_list)
    f1_macro = f1_score(test_correct_labels_list, submission_labels_list, average='macro')
    f1_weighted = f1_score(test_correct_labels_list, submission_labels_list, average='weighted')
    
    sum_accuracy += accuracy
    sum_f1_macro += f1_macro
    sum_f1_weighted += f1_weighted
    
    if accuracy > 0.66 and f1_macro > 0.66 and f1_weighted > 0.66:
        print('Accuracy Score : {}'.format(accuracy))
        print('F1 Score (Macro) : {}'.format(f1_macro))
        print('F1 Score (Weighted) : {}'.format(f1_weighted))
        print()

avg_accuracy = sum_accuracy / 5.0
avg_f1_macro = sum_f1_macro / 5.0
avg_f1_weighted = sum_f1_weighted / 5.0

print('Average Accuracy Score : {}'.format(avg_accuracy))
print('Average F1 Score (Macro) : {}'.format(avg_f1_macro))
print('Average F1 Score (Weighted) : {}'.format(avg_f1_weighted))
print()