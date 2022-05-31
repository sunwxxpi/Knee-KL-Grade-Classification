import cv2
import numpy as np
import os

image_path = './KneeXray/test/4/'
image_path2 = './KneeXray/test/4_cn/'
img_list = os.listdir(image_path)

for i in img_list:
    img = cv2.imread('{}{}'.format(image_path, i), cv2.IMREAD_GRAYSCALE)
    
    img = cv2.normalize(img, None, 15, 255, cv2.NORM_MINMAX)
    min = np.average(img.flatten())
    img = cv2.subtract(img, min-160) # -min+160

    cv2.imwrite('{}{}'.format(image_path2, i), img)