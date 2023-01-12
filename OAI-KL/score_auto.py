import os
import argparse
import natsort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action="store")
args = parser.parse_args()

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_correct_labels = test_csv['label']
test_correct_labels_list = test_correct_labels.values.tolist()

image_size_dir = (args.image_size, args.image_size)
submission_path = './submission/{}/{}/'.format(args.model_type, image_size_dir)
submission_list = os.listdir(submission_path)
submission_list_csv = [file for file in submission_list if file.endswith(".csv")]
submission_list_csv = natsort.natsorted(submission_list_csv)
submission_num = len(submission_list_csv)

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
    accuracy = round(accuracy, 4)
    f1_macro = round(f1_macro, 4)
    f1_weighted = round(f1_weighted, 4)
    
    if accuracy < 0.67 or f1_macro < 0.67 or f1_weighted < 0.67:
        print('Accuracy Score : {}'.format(accuracy))
        print('F1 Score (Macro) : {}'.format(f1_macro))
        # print('F1 Score (Weighted) : {}'.format(f1_weighted))
        os.remove('{}{}'.format(submission_path, i))
        print('{}{} Removed'.format(submission_path, i))
        print()

    elif accuracy > 0.67 and f1_macro > 0.67 and f1_weighted > 0.67:
        print('Accuracy Score : {}'.format(accuracy))
        print('F1 Score (Macro) : {}'.format(f1_macro))
        # print('F1 Score (Weighted) : {}'.format(f1_weighted))
        print()
        
avg_accuracy = sum_accuracy / submission_num
avg_f1_macro = sum_f1_macro / submission_num
avg_f1_weighted = sum_f1_weighted / submission_num
avg_accuracy = round(avg_accuracy, 4)
avg_f1_macro = round(avg_f1_macro, 4)
avg_f1_weighted = round(avg_f1_weighted, 4)

print('-------------------------------')
print('{}'.format(submission_path))
print('Average Accuracy Score : {}'.format(avg_accuracy))
print('Average F1 Score (Macro) : {}'.format(avg_f1_macro))
# print('Average F1 Score (Weighted) : {}'.format(avg_f1_weighted))
print()