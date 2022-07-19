import cv2
import os

image_path = './KneeXray/test/4/'
image_path2 = './KneeXray/test/4_clahe/'
img_list = os.listdir(image_path)

for i in img_list:
    img = cv2.imread('{}{}'.format(image_path, i), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)

    cv2.imwrite('{}{}'.format(image_path2, i), img2)