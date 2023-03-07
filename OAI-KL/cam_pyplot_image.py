import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

xlabels = ['xlabel', 'Original', 'DenseNet-161', 'EfficientNet-b5', 'EfficientNet-V2-s', 'RegNet-Y-8GF', 'ResNet-101', 'ResNext-50-32x4d', 'Wide-ResNet-50-2', 'ShuffleNet-V2-x2-0', 'CAM Ensemble']
    
def make_cam_pyplot(class_num):
    fig = plt.figure(figsize=(6, 12), dpi=200) # rows*cols 행렬의 pos번째 subplot 생성
    rows = 5
    cols = 2

    for i in tqdm(img_list, desc=f'Class {class_num}', unit='Images'):
        for j, pos in zip(img_dir_list, range(1, 11)):
            img = cv2.imread(f"{j}/{i}", cv2.IMREAD_COLOR)
            ax = fig.add_subplot(rows, cols, pos)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xlabel(xlabels[pos])
            ax.set_xticks([]), ax.set_yticks([])
            
            if pos == 10:
                ax.xaxis.label.set_color('red')
            
        plt.savefig(f"{save_dir}/{i}")
        
for class_num in range(0, 5):
    original_img_dir = f'./KneeXray/test/{class_num}'
    densenet_161_cam_dir = f'./Grad CAM/densenet_161/{class_num}'
    efficientnet_b5_cam_dir = f'./Grad CAM/efficientnet_b5/{class_num}'
    efficientnet_v2_s_cam_dir = f'./Grad CAM/efficientnet_v2_s/{class_num}'
    regnet_y_8gf_cam_dir = f'./Grad CAM/regnet_y_8gf/{class_num}'
    resnet_101_cam_dir = f'./Grad CAM/resnet_101/{class_num}'
    resnext_50_32x4d_cam_dir = f'./Grad CAM/resnext_50_32x4d/{class_num}'
    wide_resnet_50_2_cam_dir = f'./Grad CAM/wide_resnet_50_2/{class_num}'
    shufflenet_v2_x2_0_cam_dir = f'./Grad CAM/shufflenet_v2_x2_0/{class_num}'
    ensemble_cam_dir = f'./Grad CAM/ensemble_cam/{class_num}'

    save_dir = f'./Grad CAM/pyplot/{class_num}'
    
    img_list = os.listdir(original_img_dir)
    img_dir_list = [
        original_img_dir,
        densenet_161_cam_dir,
        efficientnet_b5_cam_dir,
        efficientnet_v2_s_cam_dir,
        regnet_y_8gf_cam_dir,
        resnet_101_cam_dir,
        resnext_50_32x4d_cam_dir,
        wide_resnet_50_2_cam_dir,
        shufflenet_v2_x2_0_cam_dir,
        ensemble_cam_dir
        ]
    
    make_cam_pyplot(class_num)