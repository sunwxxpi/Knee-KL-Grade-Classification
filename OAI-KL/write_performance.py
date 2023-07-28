import os
import argparse
import natsort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action="store")
args = parser.parse_args()

image_size_tuple = (args.image_size, args.image_size)

test_csv = pd.read_csv('./KneeXray/HH_2_center_crop/HH_2_center_crop.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

# model_path = './models'
# submission_path = './submission'
model_path = f'./models/{args.model_type}/{image_size_tuple}'
submission_path = f'./submission/{args.model_type}/{image_size_tuple}'

model_list = os.listdir(model_path)
model_list_pt = [file for file in model_list if file.endswith('.pt')]
model_list_pt = natsort.natsorted(model_list_pt)

submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith('.csv')]
submission_list_csv = natsort.natsorted(submission_list_csv)

submission_num = len(submission_list_csv)

sum_accuracy = 0
sum_f1_macro = 0
sum_f1_weighted = 0

for model, submission in zip(model_list_pt, submission_list_csv):
    submission_csv = pd.read_csv(f'{submission_path}/{submission}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    submission_labels = submission_csv['label']
    submission_labels_list = submission_labels.values.tolist()
    
    accuracy = accuracy_score(test_correct_labels_list, submission_labels_list)
    f1_macro = f1_score(test_correct_labels_list, submission_labels_list, average='macro')
    f1_weighted = f1_score(test_correct_labels_list, submission_labels_list, average='weighted')
    
    with open('./performance.txt', 'a', encoding='utf8') as f:
        f.write(f"{os.path.splitext(model)[0]} Accuracy Score, F1 Score (Macro, Weighted) : {accuracy:.4f}, {f1_macro:.4f}, {f1_weighted:.4f}\n")
    
    sum_accuracy += accuracy
    sum_f1_macro += f1_macro
    sum_f1_weighted += f1_weighted
    
avg_accuracy = sum_accuracy / submission_num
avg_f1_macro = sum_f1_macro / submission_num
avg_f1_weighted = sum_f1_weighted / submission_num

with open('./performance.txt', 'a', encoding='utf8') as f:
    f.write('-------------------------------\n')
    f.write('Setting : \n\n')
    f.write(f'Average Accuracy Score : {avg_accuracy:.4f}\n')
    f.write(f'Average F1 Score (Macro) : {avg_f1_macro:.4f}\n')
    f.write(f'Average F1 Score (Weighted) : {avg_f1_weighted:.4f}\n')
    f.write('-------------------------------\n\n\n')
    
print("Write Performance at 'performance.txt'")