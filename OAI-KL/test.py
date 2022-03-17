import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torch import nn, optim
from torchvision import models
import torch.nn.functional as F
from PIL import Image
from dataset import ImageDataset

test = pd.read_csv('./KneeXray/Test_he_correct.csv')

transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                               ])

test_data = ImageDataset(test)
testloader = DataLoader(test_data, batch_size = 1, shuffle = False)

fold = 1
epoch = 5

model_ft = torch.load('./models/kfold_CNN_{}fold_epoch{}.pt'.format(fold, epoch))
preds = []

for batch in testloader:
    with torch.no_grad():
        image = batch['image'].cuda()
        output = model_ft(image)
        preds.extend([i.item() for i in torch.argmax(output, axis = 1)])

submit = pd.DataFrame({'data':[i.split('/')[-1] for i in test['data']], 'label':preds})
submit.to_csv('./submission/{}fold_epoch{}_submission.csv'.format(fold, epoch), index = False)
print('saving submission.csv.....')