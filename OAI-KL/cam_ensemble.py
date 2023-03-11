import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
from torch import nn
from torchvision import models
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, FullGrad
from PIL import Image
from tqdm import tqdm

def cam_image(image, image_size_tuple, cam):
    image = image.resize(image_size_tuple)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Max Min normalization
    convert_tensor = torchvision.transforms.ToTensor()
    input_tensor = convert_tensor(image).unsqueeze(0).float()

    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :]

    img = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    
    return img

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_img = test_csv['data']
test_img_list = test_img.values.tolist()

cam_list = []

for i in tqdm(test_img_list, unit='Images'):
    image = Image.open(i).convert('RGB')
    
    model_ft = models.densenet161(weights='DEFAULT')
    in_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(in_ftrs, 5)

    model_ft.load_state_dict(torch.load('./Grad CAM/1_DenseNet-161.pt'))
    model_ft.eval()

    cam = GradCAM(model=model_ft, target_layers=[model_ft.features[-1]], use_cuda=True)
    
    image_size_tuple = (456, 456)
    cam_1 = cam_image(image, image_size_tuple, cam)
    
    model_ft = models.efficientnet_b5(weights='DEFAULT')
    in_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/2_EfficientNet-b5.pt'))
    model_ft.eval()
    
    image_size_tuple = (456, 456)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.features[-1]], use_cuda=True)
    
    cam_2 = cam_image(image, image_size_tuple, cam)
    
    model_ft = models.efficientnet_v2_s(weights='DEFAULT')
    in_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/3_EfficientNet-V2-s.pt'))
    model_ft.eval()
    
    image_size_tuple = (384, 384)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.features[-1]], use_cuda=True)
    
    cam_3 = cam_image(image, image_size_tuple, cam)
    
    model_ft = models.regnet_y_8gf(weights='DEFAULT')
    in_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/4_RegNet-Y-8GF.pt'))
    model_ft.eval()
    
    image_size_tuple = (448, 448)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.trunk_output[-1]], use_cuda=True)
    
    cam_4 = cam_image(image, image_size_tuple, cam)

    model_ft = models.resnet101(weights='DEFAULT')
    in_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/5_ResNet-101.pt'))
    model_ft.eval()
    
    image_size_tuple = (456, 456)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.layer4[-1]], use_cuda=True)

    cam_5 = cam_image(image, image_size_tuple, cam)
    
    model_ft = models.resnext50_32x4d(weights='DEFAULT')
    in_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/6_ResNext-50-32x4d.pt'))
    model_ft.eval()
    
    image_size_tuple = (512, 512)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.layer4[-1]], use_cuda=True)

    cam_6 = cam_image(image, image_size_tuple, cam)
    
    model_ft = models.wide_resnet50_2(weights='DEFAULT')
    in_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/7_Wide-ResNet-50-2.pt'))
    model_ft.eval()
    
    image_size_tuple = (456, 456)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.layer4[-1]], use_cuda=True)

    cam_7 = cam_image(image, image_size_tuple, cam)
    
    model_ft = models.shufflenet_v2_x2_0(weights='DEFAULT')
    in_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_ftrs, 5)
    
    model_ft.load_state_dict(torch.load('./Grad CAM/8_ShuffleNet-V2-x2-0.pt'))
    model_ft.eval()

    image_size_tuple = (512, 512)
    cam = GradCAM(model=model_ft, target_layers=[model_ft.conv5], use_cuda=True)

    cam_8 = cam_image(image, image_size_tuple, cam)

    ensemble_cam = (cam_1 + cam_2 + cam_3 + cam_4 + cam_5 + cam_6 + cam_7 + cam_8) / 8
    
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Max Min normalization
    cam = 0.65 * ensemble_cam + 0.35 * image
    cam = cam / np.max(cam)
    visualization = np.uint8(255 * cam)
    visualization = Image.fromarray(visualization, 'RGB')
    
    visualization.save(f"./Grad CAM/ensemble_cam/{i.split(f'test/')[-1]}")