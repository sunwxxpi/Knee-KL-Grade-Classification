import os
import argparse
import natsort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action="store")
parser.add_argument('-t', '--threshold', type=float, default=0.65, dest='threshold', action="store")
parser.add_argument('-r', '--remove_option', default=False, dest='remove_option', action="store_true")
args = parser.parse_args()

image_size_tuple = (args.image_size, args.image_size)

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

submission_path = './submission'
# submission_path = f'./submission/{args.model_type}/{image_size_tuple}'
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith('.csv')]
submission_list_csv = natsort.natsorted(submission_list_csv)
submission_num = len(submission_list_csv)

sum_accuracy = 0
sum_f1_macro = 0
sum_f1_weighted = 0

for i in submission_list_csv:
    submission_csv = pd.read_csv(f'{submission_path}/{i}', names=['data', 'label', 'prob_correct', 'prob_predict', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4'], skiprows=1)
    submission_labels = submission_csv['label']
    submission_labels_list = submission_labels.values.tolist()
    
    accuracy = accuracy_score(test_correct_labels_list, submission_labels_list)
    f1_macro = f1_score(test_correct_labels_list, submission_labels_list, average='macro')
    f1_weighted = f1_score(test_correct_labels_list, submission_labels_list, average='weighted')
    
    sum_accuracy += accuracy
    sum_f1_macro += f1_macro
    sum_f1_weighted += f1_weighted
    accuracy = round(accuracy, 4)
    f1_macro = round(f1_macro, 4)
    f1_weighted = round(f1_weighted, 4)

    print(f'{i}')
    
    if accuracy >= args.threshold and f1_macro >= args.threshold and f1_weighted >= args.threshold:
        print(f"Accuracy Score : {accuracy}")
        print(f"F1 Score (Macro) : {f1_macro}")
        # print(f"F1 Score (Weighted) : {f1_weighted}")
        # print(f"Accuracy Score + F1 Score (Macro) : {accuracy + f1_macro}")
        print()
    elif accuracy < args.threshold or f1_macro < args.threshold or f1_weighted < args.threshold:
        if args.remove_option == True:
            os.remove(f'{submission_path}/{i}')
            print(f'{submission_path}/{i} Removed')
            print()
        
avg_accuracy = sum_accuracy / submission_num
avg_f1_macro = sum_f1_macro / submission_num
avg_f1_weighted = sum_f1_weighted / submission_num
avg_accuracy = round(avg_accuracy, 4)
avg_f1_macro = round(avg_f1_macro, 4)
avg_f1_weighted = round(avg_f1_weighted, 4)

print('-------------------------------')
print(f'{submission_path}')
print(f"Average Accuracy Score : {avg_accuracy}")
print(f"Average F1 Score (Macro) : {avg_f1_macro}")
# print("Average F1 Score (Weighted) : {avg_f1_weighted}")
print()