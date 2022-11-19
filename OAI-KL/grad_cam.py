import torch
import torchvision
import numpy as np
import pandas as pd
from torch import nn
from torchvision import models
from pytorch_grad_cam import AblationCAM, EigenCAM, FullGrad, GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

""" model_ft = models.densenet201()
in_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(in_ftrs, 5) """

# model_ft = models.efficientnet_b5()
model_ft = models.efficientnet_v2_s()
in_ftrs = model_ft.classifier._modules.__getitem__('1').__getattribute__('in_features')
sequential_0 = model_ft.classifier._modules.get('0')
sequential_1 = nn.Linear(in_ftrs, 5)
model_ft.classifier = nn.Sequential(sequential_0, sequential_1)

# model_ft = torch.load('./models/DenseNet-201/models_448_DenseNet-201_32_0.0006/kfold_CNN_1fold_epoch18.pt') # DenseNet-201 Best
# model_ft.load_state_dict(torch.load('./models/kfold_CNN_2fold_epoch26.pt')) # DenseNet-201 CAM
# model_ft.load_state_dict(torch.load('./models/EfficientNet-b5/models_456_EfficientNet-b5_16_0.0005/kfold_CNN_5fold_epoch13.pt')) # EfficientNet-b5 Best
# model_ft.load_state_dict(torch.load('./models/kfold_CNN_2fold_epoch4.pt')) # EfficientNet-b5 CAM
model_ft.load_state_dict(torch.load('./models/EfficientNet-V2-s/models_384_EfficientNet-V2-s_16_0.0007/kfold_CNN_5fold_epoch35.pt')) # EfficientNet-V2-s Best & CAM
model_ft.eval()

# target_layers = [model_ft.module.features[-1]]
target_layers = [model_ft.features[-1]]

# cam = GradCAM(model=model_ft, target_layers=target_layers, use_cuda=True)
# cam = GradCAMPlusPlus(model=model_ft, target_layers=target_layers, use_cuda=True)
cam = ScoreCAM(model=model_ft, target_layers=target_layers, use_cuda=True)

test_csv = pd.read_csv('./KneeXray/Test_correct.csv', names=['data', 'label'], skiprows=1)
test_img = test_csv['data']
test_img_list = test_img.values.tolist()

# path = ['./KneeXray/test/0/9850534R.png']

for i in test_img_list:
    # Note: input_tensor can be a batch tensor with several images! """
    image = Image.open(i).convert('RGB')
    # image = image.resize((448, 448))
    # image = image.resize((456, 456))
    image = image.resize((384, 384))
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
    cam_img.save('C:/Users/PiSunWoo-RTOS/Desktop/KLGrade Project/Final/Grad CAM/{}'.format(i.split('test')[-1]))

""" save = input('If you want to save, please enter \'y\' or \'Y\'').lower()
if save == 'y':
    cam_img.save('C:/Users/PiSunWoo-RTOS/Desktop/{}'.format(path.split('/')[-1]))
    print("Image saved at \'C:/Users/PiSunWoo-RTOS/Desktop/{}\'".format(path.split('/')[-1]))
else:
    print("Image not saved.") """