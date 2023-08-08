import os
import argparse

import pandas as pd

import natsort
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action="store")
parser.add_argument('-t', '--threshold', type=float, default=0.65, dest='threshold', action="store")
parser.add_argument('-r', '--remove_option', default=False, dest='remove_option', action="store_true")
args = parser.parse_args()

image_size_tuple = (args.image_size, args.image_size)

test_csv = pd.read_csv('./KneeXray/test/test_correct.csv', names=['data', 'label'], skiprows=1)
# test_csv = pd.read_csv('./KneeXray/HH_2/HH_2.csv', names=['data', 'label'], skiprows=1)

test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

# submission_path = './submission'
basic_path = f'{args.model_type}/{image_size_tuple}'
submission_path = os.path.join('./submission', basic_path)

submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith('.csv')]
submission_list_csv = natsort.natsorted(submission_list_csv)

submission_num = len(submission_list_csv)

sum_accuracy = 0
sum_f1_macro = 0
sum_f1_weighted = 0

for fold in range(1, 6):
    fold_sum_accuracy = 0
    fold_sum_f1_macro = 0
    fold_sum_f1_weighted = 0
    
    fold_submission_list_csv = [file for file in submission_list_csv if file.startswith(f'{fold}fold')]

    fold_submission_num = len(fold_submission_list_csv)

    for submission in fold_submission_list_csv:
        submission_csv = pd.read_csv(f'{submission_path}/{submission}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
        submission_labels = submission_csv['label']
        submission_labels_list = submission_labels.values.tolist()
        
        accuracy = accuracy_score(test_correct_labels_list, submission_labels_list)
        f1_macro = f1_score(test_correct_labels_list, submission_labels_list, average='macro')
        f1_weighted = f1_score(test_correct_labels_list, submission_labels_list, average='weighted')
        
        fold_sum_accuracy += accuracy
        fold_sum_f1_macro += f1_macro
        fold_sum_f1_weighted += f1_weighted
        
        sum_accuracy += accuracy
        sum_f1_macro += f1_macro
        sum_f1_weighted += f1_weighted
        
        if accuracy >= args.threshold and f1_macro >= args. threshold and f1_weighted >= args.threshold:
            print(f"{submission.split('_submission')[0]} Accuracy Score, F1 Score (Macro, Weighted) : {accuracy:.4f}, {f1_macro:.4f}, {f1_weighted:.4f}")
            print()
        else:
            print(f"{submission.split('_submission')[0]}")
            print()
            
    fold_avg_accuracy = fold_sum_accuracy / fold_submission_num
    fold_avg_f1_macro = fold_sum_f1_macro / fold_submission_num
    fold_avg_f1_weighted = fold_sum_f1_weighted / fold_submission_num
    
    print(f'----------------{fold} Fold Average Score----------------')
    print(f'{basic_path}: {fold} Fold')
    print(f"Fold Average Accuracy Score : {fold_avg_accuracy:.4f}")
    print(f"Fold Average F1 Score (Macro) : {fold_avg_f1_macro:.4f}")
    print(f"Fold Average F1 Score (Weighted) : {fold_avg_f1_weighted:.4f}")
    print(f'----------------{fold} Fold Average Score----------------')
    print()
    
    if args.remove_option:
        print('----------------Remove Process----------------')
        for submission in fold_submission_list_csv:
            submission_csv = pd.read_csv(f'{submission_path}/{submission}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
            submission_labels = submission_csv['label']
            submission_labels_list = submission_labels.values.tolist()
            
            accuracy = accuracy_score(test_correct_labels_list, submission_labels_list)
            f1_macro = f1_score(test_correct_labels_list, submission_labels_list, average='macro')
            f1_weighted = f1_score(test_correct_labels_list, submission_labels_list, average='weighted')
            
            if accuracy < fold_avg_accuracy and f1_macro < fold_avg_f1_macro and f1_weighted < fold_avg_f1_weighted:
                print(f"Accuracy Score, F1 Score (Macro, Weighted) : {accuracy:.4f}, {f1_macro:.4f}, {f1_weighted:.4f}")
                
                os.remove(f'./models/{basic_path}/{submission.split("_submission")[0]}.pt')
                os.remove(f'{submission_path}/{submission}')
                print(f'{submission.split("_submission")[0]} Model, Submission Removed')
        print('----------------Remove Process----------------')
        print()
    
avg_accuracy = sum_accuracy / submission_num
avg_f1_macro = sum_f1_macro / submission_num
avg_f1_weighted = sum_f1_weighted / submission_num
            
print('----------------Total Average Score----------------')
print(f'{basic_path}: Total')
print(f"Total Average Accuracy Score : {avg_accuracy:.4f}")
print(f"Total Average F1 Score (Macro) : {avg_f1_macro:.4f}")
print(f"Total Average F1 Score (Weighted) : {avg_f1_weighted:.4f}")
print('----------------Total Average Score----------------')