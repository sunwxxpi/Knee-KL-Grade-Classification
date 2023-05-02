import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def calculate_performace(model_combinations_num):
    submission_csv = pd.read_csv(f'{submission_ensemble_all_dir}/{model_combinations_dir}/{submission}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    submission_labels = submission_csv['label']
    submission_labels_list = submission_labels.values.tolist()
    
    accuracy = accuracy_score(test_correct_labels_list, submission_labels_list)
    f1_macro = f1_score(test_correct_labels_list, submission_labels_list, average='macro')
    
    globals()[f'performance_{model_combinations_num}'].append(round(accuracy, 4))
    # globals()[f'performance_{model_combinations_num}'].append(round(f1_macro, 4))

for i in range(1, 9):
    globals()[f'performance_8c{i}'] = []
for i in range(1, 9):
    globals()[f'best_performance_8c{i}'] = []

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

submission_ensemble_all_dir = './Ensemble Network Box Plot/all'
model_combinations_dir_list = os.listdir(submission_ensemble_all_dir)

for model_combinations_dir_index, model_combinations_dir in enumerate(model_combinations_dir_list, start=1):
    model_combinations_list = os.listdir(f'{submission_ensemble_all_dir}/{model_combinations_dir}')
    submission_list_csv = [file for file in model_combinations_list if file.endswith('.csv')]

    for index, submission in enumerate(submission_list_csv, start=1):
        calculate_performace(model_combinations_num=model_combinations_dir)
        
        if model_combinations_dir_index != 1 and index % 3 == 0:
            globals()[f'best_performance_{model_combinations_dir}'].append(max(globals()[f'performance_{model_combinations_dir}']))
            globals()[f'performance_{model_combinations_dir}'].clear()

plt.rcParams['figure.figsize'] = (6.5, 5.5)
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots()

ax.set_ylim(0.675, 0.78)
ax.set_xlabel('Number of Models in Ensemble Network', size=15, labelpad=8)
ax.set_ylabel('Accuracy', size=15, labelpad=8)
# ax.set_ylabel('F1 Score', size=15, labelpad=8)

ax.boxplot([
        globals()['performance_8c1'],
        globals()['best_performance_8c2'],
        globals()['best_performance_8c3'],
        globals()['best_performance_8c4'],
        globals()['best_performance_8c5'],
        globals()['best_performance_8c6'],
        globals()['best_performance_8c7'],
        globals()['best_performance_8c8']
        ])

plt.title('Base Ensemble Network Performance', size=18, pad=8)
# plt.title('Optimized Ensemble Network Performance', size=18, pad=8)
plt.xticks(range(1, 9), ['Single Model', 2, 3, 4, 5, 6, 7, 8])

plt.show()