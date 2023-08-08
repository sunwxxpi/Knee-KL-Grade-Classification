from torch import nn
from torchvision import models

def model_return(args):
    if args.model_type == 'resnet_101':
        model_ft = models.resnet101(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
                
    elif args.model_type == 'resnext_50_32x4d':
        model_ft = models.resnext50_32x4d(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'wide_resnet_50_2':
        model_ft = models.wide_resnet50_2(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'densenet_169':
        model_ft = models.densenet169(weights='DEFAULT')
        in_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_ftrs, 5)
    
    elif args.model_type == 'densenet_201':
        model_ft = models.densenet201(weights='DEFAULT')
        in_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'densenet_161':
        model_ft = models.densenet161(weights='DEFAULT')
        in_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'efficientnet_b3':
        model_ft = models.efficientnet_b3(weights='DEFAULT')
        in_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'efficientnet_b5':
        model_ft = models.efficientnet_b5(weights='DEFAULT')
        in_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'efficientnet_v2_s':
        model_ft = models.efficientnet_v2_s(weights='DEFAULT')
        in_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
        
        # for num, block_layer in enumerate(model_ft.features.children()):
        #     while num < 6:
        #         for layer in block_layer:
        #             for param in layer.parameters():
        #                 param.requires_grad = False
        #         break

    elif args.model_type == 'regnet_y_8gf':
        model_ft = models.regnet_y_8gf(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'shufflenet_v2_x2_0':
        model_ft = models.shufflenet_v2_x2_0(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
        
    elif args.model_type == 'inception_v3':
        model_ft = models.inception_v3(weights='DEFAULT')
        in_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_ftrs, 5)
    
    elif args.model_type == 'PingJunChen_vgg_19':
        model_ft = models.vgg19(weights='DEFAULT')
        in_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(in_ftrs, 5)
        
    return model_ft