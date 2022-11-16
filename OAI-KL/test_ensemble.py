import os
import torch
import numpy as np
import pandas as pd

def probs_to_csv(probs_ensemble, set_epoch, mode):
    probs_ensemble = np.array(probs_ensemble)
    probs_0 = probs_ensemble[:, 0]
    probs_1 = probs_ensemble[:, 1]
    probs_2 = probs_ensemble[:, 2]
    probs_3 = probs_ensemble[:, 3]
    probs_4 = probs_ensemble[:, 4]

    probs_correct = [0 for i in range(test_image_num)]
    probs_predict = [0 for i in range(test_image_num)]
                
    for i in range(test_image_num):
        if test_correct_labels_list[i] == 0:
            probs_correct[i] = probs_0[i]
        elif test_correct_labels_list[i] == 1:
            probs_correct[i] = probs_1[i]
        elif test_correct_labels_list[i] == 2:
            probs_correct[i] = probs_2[i]
        elif test_correct_labels_list[i] == 3:
            probs_correct[i] = probs_3[i]
        elif test_correct_labels_list[i] == 4:
            probs_correct[i] = probs_4[i]
            
    for i in range(test_image_num):
        if preds[i] == 0:
            probs_predict[i] = probs_0[i]
        elif preds[i] == 1:
            probs_predict[i] = probs_1[i]
        elif preds[i] == 2:
            probs_predict[i] = probs_2[i]
        elif preds[i] == 3:
            probs_predict[i] = probs_3[i]
        elif preds[i] == 4:
            probs_predict[i] = probs_4[i]
            
    submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test_csv['data']], 'label':preds, 'prob_correct':probs_correct, 'prob_predict':probs_predict, 'prob_0':probs_0, 'prob_1':probs_1, 'prob_2':probs_2, 'prob_3':probs_3, 'prob_4':probs_4})

    submit.to_csv('{}10fold_epoch{}_submission.csv'.format(submission_path, set_epoch), index=False)
    print('save {} ensemble_submission.csv'.format(mode))
    
    
def hard_voting(probs_ensemble): # Hard Voting
    global preds
    preds = []
    
    for i in range(test_image_num):
        if len(labels_ensemble[i]) == len(set(labels_ensemble[i])):
            preds.append(min(labels_ensemble[i])) # Select [Minimum Class]
        else:
            preds.append(max(set(labels_ensemble[i]), key=labels_ensemble[i].count))
            
    probs_to_csv(probs_ensemble=probs_ensemble, set_epoch=10, mode='hard')
    
def soft_voting(probs_ensemble): # Soft Voting
    global preds
    preds = []
    
    ensemble_output = torch.tensor(probs_ensemble)
    preds.extend([i.item() for i in torch.argmax(ensemble_output, axis=1)])
    
    probs_to_csv(probs_ensemble=probs_ensemble, set_epoch=11, mode='soft')
            
def mix_voting(probs_ensemble): # Hard Voting + Soft Voting
    global preds
    preds = []
    
    for i in range(test_image_num):
        if len(labels_ensemble[i]) == len(set(labels_ensemble[i])): # Soft Voting
            probs_ensemble_output = torch.tensor([probs_ensemble[i]])
            preds.extend([j.item() for j in torch.argmax(probs_ensemble_output, axis=1)])
        else: # Hard Voting
            preds.append(max(set(labels_ensemble[i]), key=labels_ensemble[i].count))
            
    probs_to_csv(probs_ensemble=probs_ensemble, set_epoch=12, mode='mix')

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

test_image_num = len(test_correct_labels_list)

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]

labels_ensemble = [[0 for j in range(3)] for i in range(test_image_num)]
probs_ensemble = [[0 for j in range(5)] for i in range(test_image_num)]

submission_index = 0
for i in submission_list_csv:
    submission_csv = pd.read_csv('{}{}'.format(submission_path, i), names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    
    submission_labels = submission_csv['label']
    globals()['{}_labels'.format(i)] = submission_labels.values.tolist()
    globals()['{}_probs'.format(i)] = []
    
    for j in range(test_image_num):
        labels_ensemble[j][submission_index] = globals()['{}_labels'.format(i)][j]
    submission_index += 1
    
    for j in range(test_image_num):
        globals()['{}_image_{}'.format(i, j)] = [0 for k in range(5)]
        
    for j in range(5):
        submission_probs = submission_csv['prob_{}'.format(j)]
        submission_probs_list = submission_probs.values.tolist()
        
        for k in range(test_image_num):
            globals()['{}_image_{}'.format(i, k)][j] = submission_probs_list[k]
            
    for j in range(test_image_num):
        globals()['{}_probs'.format(i)].append(globals()['{}_image_{}'.format(i, j)])

""" # 수동 Ensemble : 가중치 조절 가능
for i in range(test_image_num):
    for j in range(5):
        # probs_ensemble[i][j] = (globals()['{}_probs'.format('2fold_epoch7_submission.csv')][i][j] + globals()['{}_probs'.format('5fold_epoch23_submission.csv')][i][j]) / 2
            
        probs_ensemble[i][j] = (globals()['{}_probs'.format('5fold_epoch30_submission.csv')][i][j] + globals()['{}_probs'.format('2fold_epoch5_submission.csv')][i][j] + globals()['{}_probs'.format('5fold_epoch19_submission.csv')][i][j]) / 3 """

# 자동 Ensemble : 가중치 1로 고정
for i in submission_list_csv:
    for j in range(test_image_num):
        for k in range(5):
            probs_ensemble[j][k] += globals()['{}_probs'.format(i)][j][k]
                                 
# probs_ensemble[j][k] = probs_ensemble[j][k] / 2
probs_ensemble[j][k] = probs_ensemble[j][k] / 3

hard_voting(probs_ensemble)
soft_voting(probs_ensemble)
mix_voting(probs_ensemble)