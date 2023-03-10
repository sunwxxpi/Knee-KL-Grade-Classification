import argparse
import torch
import torchvision
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from tqdm import tqdm
from model import model_return

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', dest='model_type', action='store')
args = parser.parse_args()

model_ft = model_return(args)
if args.model_type == 'densenet_161':
    model_ft.load_state_dict(torch.load('./Grad CAM/1_DenseNet-161.pt'))
    target_layers = [model_ft.features[-1]]
    image_size_tuple = (456, 456)
    
elif args.model_type == 'efficientnet_b5':
    model_ft.load_state_dict(torch.load('./Grad CAM/2_EfficientNet-b5.pt'))
    target_layers = [model_ft.features[-1]]
    image_size_tuple = (456, 456)
    
elif args.model_type == 'efficientnet_v2_s':
    model_ft.load_state_dict(torch.load('./Grad CAM/3_EfficientNet-V2-s.pt'))
    target_layers = [model_ft.features[-1]]
    image_size_tuple = (384, 384)
    
elif args.model_type == 'regnet_y_8gf':
    model_ft.load_state_dict(torch.load('./Grad CAM/4_RegNet-Y-8GF.pt'))
    target_layers = [model_ft.trunk_output[-1]]
    image_size_tuple = (448, 448)
    
elif args.model_type == 'resnet_101':
    model_ft.load_state_dict(torch.load('./Grad CAM/5_ResNet-101.pt'))
    target_layers = [model_ft.layer4[-1]]
    image_size_tuple = (456, 456)
    
elif args.model_type == 'resnext_50_32x4d':
    model_ft.load_state_dict(torch.load('./Grad CAM/6_ResNext-50-32x4d.pt'))
    target_layers = [model_ft.layer4[-1]]
    image_size_tuple = (512, 512)
    
elif args.model_type == 'wide_resnet_50_2':
    model_ft.load_state_dict(torch.load('./Grad CAM/7_Wide-ResNet-50-2.pt'))
    target_layers = [model_ft.layer4[-1]]
    image_size_tuple = (456, 456)
    
elif args.model_type == 'shufflenet_v2_x2_0':
    model_ft.load_state_dict(torch.load('./Grad CAM/8_ShuffleNet-V2-X2-0.pt'))
    target_layers = [model_ft.conv5]    
    image_size_tuple = (512, 512)

model_ft.eval()

cam = GradCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = GradCAMPlusPlus(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = XGradCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = AblationCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = ScoreCAM(model=model_ft, target_layers=target_layers, use_cuda=True)

test_csv = pd.read_csv(f'./KneeXray/Test_correct_{image_size_tuple}.csv', names=['data', 'label'], skiprows=1)
test_img = test_csv['data']
test_img_list = test_img.values.tolist()

for i in tqdm(test_img_list, unit='Images'):
    # Note: input_tensor can be a batch tensor with several images! """
    image = Image.open(i).convert('RGB')
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Max Min normalization
    convert_tensor = torchvision.transforms.ToTensor()
    input_tensor = convert_tensor(image).unsqueeze(0).float()

    # We have to specify the target we want to generate the Class Activation Maps for.
    # If targets is None, the highest scoring category will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets That are, for example, combinations of categories, or specific outputs in a non standard model_ft.
    # targets = [ClassifierOutputTarget(281)]
    # target_category = None

    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :] # In this example grayscale_cam has only one image in the batch:
    
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

    cam_img = Image.fromarray(visualization, 'RGB')
    cam_img = cam_img.resize((224, 224))
    cam_img.save(f"./Grad CAM/{args.model_type}/{i.split(f'test_{image_size_tuple}/')[-1]}")