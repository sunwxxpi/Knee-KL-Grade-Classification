'''
# Histogram Equalization
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./KneeXray/train/0/9003126L.png')
hist, bins = np.histogram(img.flatten(), 256,[0,256])
cdf = hist.cumsum()

# cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
# mask처리가 되면 Numpy 계산에서 제외가 됨
# 아래는 cdf array에서 값이 0인 부분을 mask처리함
cdf_m = np.ma.masked_equal(cdf,0)

#History Equalization 공식
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

# Mask처리를 했던 부분을 다시 0으로 변환
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.subplot(122),plt.imshow(img2),plt.title('Equalization')
plt.show()
'''

'''
# CLAHE
import cv2
import numpy as np

img = cv2.imread('./KneeXray/train/0/9003126L.png', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
res = np.hstack((img,cl1)) #stacking images side-by-side

cv2.imshow('img', res)
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
# Histogram Nomalization
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 그레이 스케일로 영상 읽기
img = cv2.imread('./KneeXray/train/0/9003126L.png', cv2.IMREAD_GRAYSCALE)

#--③ OpenCV API를 이용한 정규화
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#--④ 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])


cv2.imshow('cv2.normalize()', img_norm)
cv2.waitKey()

hists = {'Before' : hist, 'cv2.normalize()' : hist_norm}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,2,i+1)
    plt.title(k)
    plt.plot(v)
plt.savefig('./KneeXray/train/0_he/9004315R.png')
'''