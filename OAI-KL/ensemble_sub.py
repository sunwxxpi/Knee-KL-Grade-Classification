import pandas as pd
import os
import torch

test_csv = pd.read_csv('./KneeXray/Test.csv')

submission_path = './submission/'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]

for i in submission_list_csv:
    submission_csv = pd.read_csv('{}{}'.format(submission_path, i), names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    
    globals()['{}_probs'.format(i)] = []
    
    for j in range(1656):
        globals()['{}_image_{}'.format(i, j)] = [0, 0, 0, 0, 0]
    
    for j in range(5):
        submission_probs = submission_csv['prob_{}'.format(j)]
        submission_probs_list = submission_probs.values.tolist()
        
        for k in range(1656):
            globals()['{}_image_{}'.format(i, k)][j] = submission_probs_list[k] 
            
    for j in range(1656):       
        globals()['{}_probs'.format(i)].append(globals()['{}_image_{}'.format(i, j)])
            
ensemble = [[0 for j in range(5)] for i in range(1656)]
preds = []

for i in range(1656):
    for k in range(5):
        # ensemble[i][k] = (globals()['{}_probs'.format('1fold_epoch14_submission.csv')][i][k] + globals()['{}_probs'.format('2fold_epoch17_submission.csv')][i][k]) / 2
    
        ensemble[i][k] = (globals()['{}_probs'.format('1fold_epoch14_submission.csv')][i][k] + globals()['{}_probs'.format('2fold_epoch17_submission.csv')][i][k] + globals()['{}_probs'.format('3fold_epoch21_submission.csv')][i][k]) / 3 # 1.35 1.35 0.3 || 1.3 1.35 0.35 || 1.35 1.3 0.35

ensemble_output = torch.tensor(ensemble)
preds.extend([i.item() for i in torch.argmax(ensemble_output, axis=1)])
    
submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test_csv['data']], 'label':preds})

submit.to_csv('{}ensemble_submission.csv'.format(submission_path), index=False)
print('save ensemble_submission.csv')