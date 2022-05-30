import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

image_path = './KneeXray/test/4/'
image_path2 = './KneeXray/test/4_he/'
img_list = os.listdir(image_path)

for i in img_list:
    img = cv2.imread('{}{}'.format(image_path, i), cv2.IMREAD_GRAYSCALE)
    hist, bins = np.histogram(img.flatten(), 256,[0,256])
    cdf = hist.cumsum()

# cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
# mask처리가 되면 Numpy 계산에서 제외가 됨
# 아래는 cdf array에서 값이 0인 부분을 mask처리함
    cdf_m = np.ma.masked_equal(cdf,0)

# History Equalization 공식
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

# Mask처리를 했던 부분을 다시 0으로 변환
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
# 이미지 저장
    img2 = cdf[img]
    cv2.imwrite('{}{}'.format(image_path2, i), img2)