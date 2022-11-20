import os
import cv2
import matplotlib.pyplot as plt

xlabels = ["xlabel", "Original", "DenseNet-201", "EfficientNet-b5", "EfficientNet-V2-s"]

original_dir = "C:/Users/PiSunWoo-RTOS/Desktop/KLGrade Project/Final/Grad CAM/Original/4/"
densenet_cam_dir = "C:/Users/PiSunWoo-RTOS/Desktop/KLGrade Project/Final/Grad CAM/DenseNet-201_ScoreCAM/4/"
efficientnet_cam_dir = "C:/Users/PiSunWoo-RTOS/Desktop/KLGrade Project/Final/Grad CAM/EfficientNet-b5_ScoreCAM/4/"
efficientnetv2_cam_dir = "C:/Users/PiSunWoo-RTOS/Desktop/KLGrade Project/Final/Grad CAM/EfficientNet-V2-s_ScoreCAM/4/"

img_dir_list = [original_dir, densenet_cam_dir, efficientnet_cam_dir, efficientnetv2_cam_dir]
img_list = os.listdir(original_dir)

save_dir = "C:/Users/PiSunWoo-RTOS/Desktop/4/"

fig = plt.figure() # rows*cols 행렬의 pos번째 subplot 생성
rows = 2
cols = 2

for i in img_list:
    pos = 1
    for j in img_dir_list:
        img = cv2.imread('{}{}'.format(j, i), cv2.IMREAD_COLOR)
        ax = fig.add_subplot(rows, cols, pos)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xlabel(xlabels[pos])
        ax.set_xticks([]), ax.set_yticks([])
        pos += 1
        
    plt.savefig(save_dir + i)