import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
from torch import nn
from torchvision import models
from pytorch_grad_cam import AblationCAM, EigenCAM, FullGrad, GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM
from PIL import Image

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_img = test_csv['data']
test_img_list = test_img.values.tolist()

for i in test_img_list:
    image = Image.open(i).convert('RGB')
    
    model_ft_1 = models.densenet201()
    in_ftrs = model_ft_1.classifier.in_features
    model_ft_1.classifier = nn.Linear(in_ftrs, 5)
    
    model_ft_1.load_state_dict(torch.load('./models/kfold_CNN_2fold_epoch26.pt'))
    model_ft_1.eval()
    cam_1 = ScoreCAM(model=model_ft_1, target_layers=[model_ft_1.features[-1]], use_cuda=True)
    
    image_1 = image.resize((448, 448))
    image_1 = (image_1 - np.min(image_1)) / (np.max(image_1) - np.min(image_1)) # Max Min normalization
    convert_tensor = torchvision.transforms.ToTensor()
    input_tensor = convert_tensor(image_1).unsqueeze(0).float()

    grayscale_cam_1 = cam_1(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
    grayscale_cam_1 = grayscale_cam_1[0, :]
    
    img_1 = cv2.applyColorMap(np.uint8(255 * grayscale_cam_1), cv2.COLORMAP_JET)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_1 = cv2.resize(img_1, (224, 224))
    img_1 = np.float32(img_1) / 255
    
    model_ft_2 = models.efficientnet_b5()
    in_ftrs = model_ft_2.classifier._modules.__getitem__('1').__getattribute__('in_features')
    in_ftrs = model_ft_2.classifier[1].in_features
    model_ft_2.classifier[1] = nn.Linear(in_ftrs, 5)
        
    model_ft_2.load_state_dict(torch.load('./models/kfold_CNN_2fold_epoch4.pt'))
    model_ft_2.eval()
    cam_2 = ScoreCAM(model=model_ft_2, target_layers=[model_ft_2.features[-1]], use_cuda=True)
    
    image_2 = image.resize((456, 456))
    image_2 = (image_2 - np.min(image_2)) / (np.max(image_2) - np.min(image_2)) # Max Min normalization
    convert_tensor = torchvision.transforms.ToTensor()
    input_tensor = convert_tensor(image_2).unsqueeze(0).float()

    grayscale_cam_2 = cam_2(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
    grayscale_cam_2 = grayscale_cam_2[0, :]
    
    img_2 = cv2.applyColorMap(np.uint8(255 * grayscale_cam_2), cv2.COLORMAP_JET)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    img_2 = cv2.resize(img_2, (224, 224))
    img_2 = np.float32(img_2) / 255
    
    model_ft_3 = models.efficientnet_v2_s()
    in_ftrs = model_ft_3.classifier._modules.__getitem__('1').__getattribute__('in_features')
    in_ftrs = model_ft_3.classifier[1].in_features
    model_ft_3.classifier[1] = nn.Linear(in_ftrs, 5)
        
    model_ft_3.load_state_dict(torch.load('./models/kfold_CNN_5fold_epoch35.pt'))
    model_ft_3.eval()
    cam_3 = ScoreCAM(model=model_ft_3, target_layers=[model_ft_3.features[-1]], use_cuda=True)
    
    image_3 = image.resize((384, 384))
    image_3 = (image_3 - np.min(image_3)) / (np.max(image_3) - np.min(image_3)) # Max Min normalization
    convert_tensor = torchvision.transforms.ToTensor()
    input_tensor = convert_tensor(image_3).unsqueeze(0).float()

    grayscale_cam_3 = cam_3(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
    grayscale_cam_3 = grayscale_cam_3[0, :]
    
    img_3 = cv2.applyColorMap(np.uint8(255 * grayscale_cam_3), cv2.COLORMAP_JET)
    img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
    img_3 = cv2.resize(img_3, (224, 224))
    img_3 = np.float32(img_3) / 255

    ensemble_cam = (img_1 + img_2 + img_3) / 3
    
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Max Min normalization
    cam = 0.6 * ensemble_cam + 0.4 * image
    cam = cam / np.max(cam)
    visualization = np.uint8(255 * cam)
    visualization = Image.fromarray(visualization, 'RGB')
    
    visualization.save(f"./Grad CAM/{i.split('test')[-1]}")