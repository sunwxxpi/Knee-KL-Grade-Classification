import cv2

image = cv2.imread('./KneeXray/train/0/9001695L.jpg')
cv2.imwrite('./KneeXray/train/9001695L.png', image)