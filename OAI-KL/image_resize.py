import cv2
import os

image_path = './KneeXray/train/4/'
image_path2 = './KneeXray/train_(512, 512)/4/'
img_list = os.listdir(image_path)

for i in img_list:
    img = cv2.imread('{}{}'.format(image_path, i), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('{}{}'.format(image_path2, i), img2)