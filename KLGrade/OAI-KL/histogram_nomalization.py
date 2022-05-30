import cv2
import numpy as np
import matplotlib.pylab as plt
import os

image_path = './KneeXray/test/4/'
image_path2 = './KneeXray/test/4_hn/'
img_list = os.listdir(image_path)

for i in img_list:
    img = cv2.imread('{}{}'.format(image_path, i), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite('{}{}'.format(image_path2, i), img2)