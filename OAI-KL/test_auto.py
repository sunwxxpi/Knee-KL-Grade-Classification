import os
import argparse
import natsort
import torch
import pandas as pd
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset
from model import model_list

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action="store")
args = parser.parse_args()
print('Model Type : {}'.format(args.model_type))
print('Image Size : ({}, {})'.format(args.image_size, args.image_size))

image_size_dir = (args.image_size, args.image_size)

test_csv = pd.read_csv('./KneeXray/Test_correct.csv')
# test_csv = pd.read_csv('./KneeXray/Test_correct_{}.csv'.format(image_size_dir))

transform = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Resize((args.image_size, args.image_size), transforms.InterpolationMode.BICUBIC),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                ])
test_data = ImageDataset(test_csv, image_size=args.image_size, transforms=transform)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

transform_tta = tta.Compose([
                            tta.HorizontalFlip()
                            ])

model_path = './models/{}/{}/'.format(args.model_type, image_size_dir)
submission_path = './submission/{}/{}/'.format(args.model_type, image_size_dir)
model_list = os.listdir(model_path)
model_list_pt = [file for file in model_list if file.endswith(".pt")]
model_list_pt = natsort.natsorted(model_list_pt)

for i in model_list_pt:
    model_ft = model_list(args)
    
    preds = []
    probs_correct = []
    probs_predict = []
    probs_0, probs_1, probs_2, probs_3, probs_4 = [], [], [], [], []

    model_ft.load_state_dict(torch.load('{}{}'.format(model_path, i)))
    # model_ft = torch.load('{}{}'.format(model_path, i))
    model_ft.eval()
    model_ft.cuda()
    model_ft = tta.ClassificationTTAWrapper(model_ft, transform_tta)
    
    for batch in testloader:
        with torch.no_grad():
            image = batch['image'].cuda()
            target = batch['target'].cuda()
            output = model_ft(image)
            preds.extend([i.item() for i in torch.argmax(output, axis=1)])
            
            softmax = nn.Softmax(dim=1)
            softmax_output = softmax(output).detach().cpu().numpy()
            probs_correct.append(softmax_output[0][int(target)])
            probs_predict.append(max(softmax_output[0]))
            for k in range(5):
                globals()['probs_{}'.format(k)].append(softmax_output[0][k])
        
    submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test_csv['data']], 'label':preds, 'prob_correct':probs_correct, 'prob_predict':probs_predict, 'prob_0':probs_0, 'prob_1':probs_1, 'prob_2':probs_2, 'prob_3':probs_3, 'prob_4':probs_4})

    fold_and_epoch = i[10:-3]
    submit.to_csv('{}{}_submission.csv'.format(submission_path, fold_and_epoch), index=False)
    print('save {}{}_submission.csv'.format(submission_path, fold_and_epoch))