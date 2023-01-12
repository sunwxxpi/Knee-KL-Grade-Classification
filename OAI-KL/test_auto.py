import os
import argparse
import natsort
import torch
import pandas as pd
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
parser.add_argument('-i', '--image_size', type=int, default=224, dest='image_size', action="store")
args = parser.parse_args()

test_csv = pd.read_csv('./KneeXray/Test_correct.csv')

transform = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Resize((args.image_size, args.image_size), transforms.InterpolationMode.BICUBIC),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                ])
test_data = ImageDataset(test_csv, image_size=args.image_size, transforms=transform)
print('Image Size : ({}, {})'.format(args.image_size, args.image_size))
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

transform_tta = tta.Compose([
                            tta.HorizontalFlip()
                            ])

image_size_dir = (args.image_size, args.image_size)
model_path = './models/{}/{}/'.format(args.model_type, image_size_dir)
submission_path = './submission/{}/{}/'.format(args.model_type, image_size_dir)
model_list = os.listdir(model_path)
model_list_pt = [file for file in model_list if file.endswith(".pt")]
model_list_pt = natsort.natsorted(model_list_pt)

print('Model Type : {}'.format(args.model_type))

for i in model_list_pt:
    if args.model_type == 'resnet_101':
        model_ft = models.resnet101(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
            
    elif args.model_type == 'densenet_169':
        model_ft = models.densenet169(weights='DEFAULT')
        in_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'efficientnet_b3':
        model_ft = models.efficientnet_b3(weights='DEFAULT')
        in_ftrs = model_ft.classifier._modules.__getitem__('1').__getattribute__('in_features')
        sequential_0 = model_ft.classifier._modules.get('0')
        sequential_1 = nn.Linear(in_ftrs, 5)
        model_ft.classifier = nn.Sequential(sequential_0, sequential_1)
        
    elif args.model_type == 'efficientnet_v2_s':
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
        in_ftrs = model_ft.classifier._modules.__getitem__('1').__getattribute__('in_features')
        sequential_0 = model_ft.classifier._modules.get('0')
        sequential_1 = nn.Linear(in_ftrs, 5)
        model_ft.classifier = nn.Sequential(sequential_0, sequential_1)
    
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