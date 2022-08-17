import torch
import torchvision
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# model = torch.load('./models/models_380_densenet-201_lr=0.0005_RandomRotation(20)_220720/kfold_CNN_2fold_epoch17.pt')
model = torch.load('./models/models_380_efficientnet-v2-s_lr=0.0007_RandomRotation(20)_220806/kfold_CNN_5fold_epoch23.pt')

target_layers = [model.module.features[-1]]
# target_layers = [model.module._bn1] # EfficientNet from lukemelas

path = "./KneeXray/test/4/9711506L.png"

# Note: input_tensor can be a batch tensor with several images! """
image = Image.open(path).convert('RGB')
image = image.resize((380, 380))
image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Max min normalization
input_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).float()

# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
# cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)

# We have to specify the target we want to generate the Class Activation Maps for.
# If targets is None, the highest scoring category will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets That are, for example, combinations of categories, or specific outputs in a non standard model.
# targets = [ClassifierOutputTarget(281)]
# target_category = None

grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=True)

grayscale_cam = grayscale_cam[0, :] # In this example grayscale_cam has only one image in the batch:
visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

cam_img = Image.fromarray(visualization, 'RGB')
cam_img = cam_img.resize((224, 224))
cam_img.show()

save = input('If you want to save, please enter \'y\' or \'Y\'').lower()
if save == 'y':
    cam_img.save('C:/Users/PiSunWoo-RTOS/Desktop/{}'.format(path.split('/')[-1]))
    print("Image saved at \'C:/Users/PiSunWoo-RTOS/Desktop/{}\'".format(path.split('/')[-1]))
else:
    print("Image not saved.")

""" 
"./KneeXray/test/0/9850534R.png"
"./KneeXray/test/0/9993442L.png"
"./KneeXray/test/1/9229496R.png"
"./KneeXray/test/1/9745578R.png"
"./KneeXray/test/2/9048192L.png"
"./KneeXray/test/2/9531901R.png"
"./KneeXray/test/3/9056326L.png"
"./KneeXray/test/3/9330729R.png"
"./KneeXray/test/4/9254422L.png"
"./KneeXray/test/4/9711506L.png"
"""