from torch import nn
from torchvision import models

def model_list(args):
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

    elif args.model_type == 'resnext':
        model_ft = models.resnext50_32x4d(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'regnet':
        model_ft = models.regnet_y_8gf(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'inception_v3':
        model_ft = models.inception_v3(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'shufflenet_v2':
        model_ft = models.shufflenet_v2_x2_0(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    """ elif args.model_type == 'convnext_tiny':
        model_ft = models.convnext_tiny(weights='DEFAULT')
        in_ftrs = model_ft.classifier._modules.__getitem__('2').__getattribute__('in_features')
        sequential_0 = model_ft.classifier._modules.get('0')
        sequential_1 = model_ft.classifier._modules.get('1')
        sequential_2 = nn.Linear(in_ftrs, 5)
        model_ft.classifier = nn.Sequential(sequential_0, sequential_1, sequential_2) """
        
    return model_ft