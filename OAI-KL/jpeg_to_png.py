import cv2
import os

image_path = './train from DPhi_jpg/'
image_path2 = './train from DPhi/'
img_list = os.listdir(image_path)

for i in img_list:
    image = cv2.imread('{}{}'.format(image_path, i), cv2.IMREAD_GRAYSCALE)

    ii = i.replace('jpg', 'png')
    cv2.imwrite('{}{}'.format(image_path2, ii), image)