import os
from collections import Counter

import numpy as np
import pandas as pd

import torch

def probs_to_csv(probs_ensemble, set_epoch, ensemble_mode):
    probs_ensemble = np.array(probs_ensemble)
    probs_0 = probs_ensemble[:, 0]
    probs_1 = probs_ensemble[:, 1]
    probs_2 = probs_ensemble[:, 2]
    probs_3 = probs_ensemble[:, 3]
    probs_4 = probs_ensemble[:, 4]

    probs_correct = []
    probs_predict = []
                
    for i in range(test_image_num):
        if test_correct_labels_list[i] == 0:
            probs_correct.append(probs_0[i])
        elif test_correct_labels_list[i] == 1:
            probs_correct.append(probs_1[i])
        elif test_correct_labels_list[i] == 2:
            probs_correct.append(probs_2[i])
        elif test_correct_labels_list[i] == 3:
            probs_correct.append(probs_3[i])
        elif test_correct_labels_list[i] == 4:
            probs_correct.append(probs_4[i])
            
    for i in range(test_image_num):
        if preds[i] == 0:
            probs_predict.append(probs_0[i])
        elif preds[i] == 1:
            probs_predict.append(probs_1[i])
        elif preds[i] == 2:
            probs_predict.append(probs_2[i])
        elif preds[i] == 3:
            probs_predict.append(probs_3[i])
        elif preds[i] == 4:
            probs_predict.append(probs_4[i])
                        
    submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test_csv['data']], 'label':preds, 'prob_correct':probs_correct, 'prob_predict':probs_predict, 'prob_0':probs_0, 'prob_1':probs_1, 'prob_2':probs_2, 'prob_3':probs_3, 'prob_4':probs_4})

    submit.to_csv(f'{submission_path}/10fold_epoch{set_epoch}_submission.csv', index=False)
    print(f"Save {ensemble_mode} Ensemble submission.csv")
    
def hard_voting(probs_ensemble): # Hard Voting
    global preds
    preds = []
    
    for i in range(test_image_num):
        counts = Counter(labels_ensemble[i])
        max_count = max(counts.values())
        
        if list(counts.values()).count(max_count) >= 2:
            most_common = [num for num, count in counts.items() if count == max_count]
            preds.append(min(most_common)) # Select [Minimum Class]
        else:
            preds.append(max(labels_ensemble[i], key=labels_ensemble[i].count))
            
    probs_to_csv(probs_ensemble=probs_ensemble, set_epoch=10, ensemble_mode='hard')
    
def soft_voting(probs_ensemble): # Soft Voting
    global preds
    preds = []
    
    ensemble_output = torch.tensor(probs_ensemble)
    preds.extend([i.item() for i in torch.argmax(ensemble_output, axis=1)])
    
    probs_to_csv(probs_ensemble=probs_ensemble, set_epoch=11, ensemble_mode='soft')
            
def mix_voting(probs_ensemble): # Hard Voting + Soft Voting = Mix Voting
    global preds
    preds = []
    
    for i in range(test_image_num):
        counts = Counter(labels_ensemble[i])
        max_count = max(counts.values())
        
        if list(counts.values()).count(max_count) >= 2: # Soft Voting
            probs_ensemble_output = torch.tensor([probs_ensemble[i]])
            preds.extend([j.item() for j in torch.argmax(probs_ensemble_output, axis=1)])
        else: # Hard Voting
            preds.append(max(labels_ensemble[i], key=labels_ensemble[i].count))
            
    probs_to_csv(probs_ensemble=probs_ensemble, set_epoch=12, ensemble_mode='mix')

test_csv = pd.read_csv('./KneeXray/test/test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

submission_path = './submission'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith('.csv')]

ensemble_model_num = len(submission_list_csv)
test_image_num = len(test_correct_labels_list)
labels_ensemble = [[0 for j in range(ensemble_model_num)] for i in range(test_image_num)]
probs_ensemble = [[0 for j in range(5)] for i in range(test_image_num)]

submission_index = 0
for i in submission_list_csv:
    submission_csv = pd.read_csv(f'{submission_path}/{i}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    
    submission_labels = submission_csv['label']
    globals()[f'{i}_labels'] = submission_labels.values.tolist()
    globals()[f'{i}_probs'] = []
    
    for j in range(test_image_num):
        labels_ensemble[j][submission_index] = globals()[f'{i}_labels'][j]
    submission_index += 1
    
    for j in range(test_image_num):
        globals()[f'{i}_image_{j}'] = [0 for k in range(5)]
        
    for j in range(5):
        submission_probs = submission_csv[f'prob_{j}']
        submission_probs_list = submission_probs.values.tolist()
        
        for k in range(test_image_num):
            globals()[f'{i}_image_{k}'][j] = submission_probs_list[k]
            
    for j in range(test_image_num):
        globals()[f'{i}_probs'].append(globals()[f'{i}_image_{j}'])

""" # 수동 Ensemble : 가중치 조절 가능
for i in range(test_image_num):
    for j in range(5):
        # probs_ensemble[i][j] = (globals()[f"{'2fold_epoch7_submission.csv'}_probs"][i][j] + globals()[f"{'5fold_epoch23_submission.csv'}_probs"][i][j]) / 2
            
        probs_ensemble[i][j] = (globals()[f"{'5fold_epoch30_submission.csv'}_probs"][i][j] + globals()[f"{'2fold_epoch5_submission.csv'}_probs"][i][j] + globals()[f"{'5fold_epoch19_submission.csv'}_probs"][i][j]) / 3 """

# 자동 Ensemble : 가중치 1로 고정
for i in submission_list_csv:
    for j in range(test_image_num):
        for k in range(5):
            probs_ensemble[j][k] += globals()[f'{i}_probs'][j][k]

probs_ensemble_array = np.array(probs_ensemble)
probs_ensemble = (probs_ensemble_array / ensemble_model_num).tolist()

hard_voting(probs_ensemble)
soft_voting(probs_ensemble)
mix_voting(probs_ensemble)