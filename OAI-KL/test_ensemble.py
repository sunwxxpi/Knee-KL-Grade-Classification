import pandas as pd
import os
import torch

test_csv = pd.read_csv('./KneeXray/Test.csv')

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]

labels_ensemble = [[0 for j in range(3)] for i in range(1656)]
probs_ensemble = [[0 for j in range(5)] for i in range(1656)]

submission_index = 0
for i in submission_list_csv:
    submission_csv = pd.read_csv('{}{}'.format(submission_path, i), names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    
    submission_labels = submission_csv['label']
    globals()['{}_labels'.format(i)] = submission_labels.values.tolist()
    globals()['{}_probs'.format(i)] = []
    
    for j in range(1656):
        labels_ensemble[j][submission_index] = globals()['{}_labels'.format(i)][j]
    submission_index += 1
    
    for j in range(1656):
        globals()['{}_image_{}'.format(i, j)] = [0 for k in range(5)]
        
    
    for j in range(5):
        submission_probs = submission_csv['prob_{}'.format(j)]
        submission_probs_list = submission_probs.values.tolist()
        
        for k in range(1656):
            globals()['{}_image_{}'.format(i, k)][j] = submission_probs_list[k] 
            
    for j in range(1656):
        globals()['{}_probs'.format(i)].append(globals()['{}_image_{}'.format(i, j)])
        
mode = 'soft_voting'
# mode = 'hard_voting'

preds = []

if mode == 'soft_voting':
    for i in range(1656):
        for k in range(5):
            # probs_ensemble[i][k] = (globals()['{}_probs'.format('1fold_epoch14_submission.csv')][i][k] + globals()['{}_probs'.format('2fold_epoch17_submission.csv')][i][k]) / 2
        
            probs_ensemble[i][k] = (globals()['{}_probs'.format('1fold_epoch53_submission.csv')][i][k] + globals()['{}_probs'.format('3fold_epoch21_submission.csv')][i][k] + globals()['{}_probs'.format('5fold_epoch4_submission.csv')][i][k]) / 3 # 1.35 1.35 0.3 || 1.3 1.35 0.35 || 1.35 1.3 0.35
    
    ensemble_output = torch.tensor(probs_ensemble)
    preds.extend([i.item() for i in torch.argmax(ensemble_output, axis=1)])
            
elif mode == 'hard_voting':
    for i in range(1656):
        if len(labels_ensemble[i]) == len(set(labels_ensemble[i])):
            for j in range(5):
                # probs_ensemble[i][j] = (globals()['{}_probs'.format('1fold_epoch14_submission.csv')][i][j] + globals()['{}_probs'.format('2fold_epoch17_submission.csv')][i][j]) / 2
            
                probs_ensemble[i][j] = (globals()['{}_probs'.format('1fold_epoch53_submission.csv')][i][j] + globals()['{}_probs'.format('3fold_epoch21_submission.csv')][i][j] + globals()['{}_probs'.format('5fold_epoch4_submission.csv')][i][j]) / 3 # 1.35 1.35 0.3 || 1.3 1.35 0.35 || 1.35 1.3 0.35
            
            probs_ensemble_output = torch.tensor([probs_ensemble[i]])
            preds.extend([j.item() for j in torch.argmax(probs_ensemble_output, axis=1)])
        
        else:
            preds.append(max(set(labels_ensemble[i]), key=labels_ensemble[i].count))
            
submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test_csv['data']], 'label':preds})

submit.to_csv('{}ensemble_submission.csv'.format(submission_path), index=False)
print('save ensemble_submission.csv')