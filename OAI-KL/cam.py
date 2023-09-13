import argparse

import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch

from model import model_return

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
args = parser.parse_args()

model_ft = model_return(args)
if args.model_type == 'densenet_161':
    model_ft.load_state_dict(torch.load('./Grad CAM/1_DenseNet-161.pt'))
    target_layers = [model_ft.features[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'efficientnet_b5':
    model_ft.load_state_dict(torch.load('./Grad CAM/2_EfficientNet-b5.pt'))
    target_layers = [model_ft.features[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'efficientnet_v2_s':
    model_ft.load_state_dict(torch.load('./Grad CAM/3_EfficientNet-V2-s.pt'))
    target_layers = [model_ft.features[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'regnet_y_8gf':
    model_ft.load_state_dict(torch.load('./Grad CAM/4_RegNet-Y-8GF.pt'))
    target_layers = [model_ft.trunk_output[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'resnet_101':
    model_ft.load_state_dict(torch.load('./Grad CAM/5_ResNet-101.pt'))
    target_layers = [model_ft.layer4[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'resnext_50_32x4d':
    model_ft.load_state_dict(torch.load('./Grad CAM/6_ResNext-50-32x4d.pt'))
    target_layers = [model_ft.layer4[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'wide_resnet_50_2':
    model_ft.load_state_dict(torch.load('./Grad CAM/7_Wide-ResNet-50-2.pt'))
    target_layers = [model_ft.layer4[-1]]
    image_size_tuple = (224, 224)
    
elif args.model_type == 'shufflenet_v2_x2_0':
    model_ft.load_state_dict(torch.load('./Grad CAM/1fold_epoch14.pt'))
    target_layers = [model_ft.conv5]    
    image_size_tuple = (224, 224)
    
model_ft.eval()

# cam = GradCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = GradCAMPlusPlus(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = XGradCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = AblationCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
cam = ScoreCAM(model=model_ft, target_layers=target_layers, use_cuda=True)

test_csv = pd.read_csv(f'./KneeXray/test/test_correct.csv', names=['data', 'label'], skiprows=1)
test_img = test_csv['data']
test_img_list = test_img.values.tolist()

for i in tqdm(test_img_list, unit='Images'):
    # Note: input_tensor can be a batch tensor with several images! """
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Max Min normalization

    transform = A.Compose([
        A.Resize(image_size_tuple[1], image_size_tuple[0], interpolation=cv2.INTER_CUBIC, p=1),
        ToTensorV2()
    ])
    convert_tensor = transform(image=image)['image']
    input_tensor = convert_tensor.unsqueeze(0).float()

    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :] # In this example grayscale_cam has only one image in the batch:
    
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

    cam_img = Image.fromarray(visualization, 'RGB')
    cam_img = cam_img.resize((224, 224))
    cam_img.save(f"./Grad CAM/Image Size Original Revision/{args.model_type}/{i.split(f'test/')[-1]}")