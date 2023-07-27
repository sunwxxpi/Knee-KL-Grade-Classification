import os
import cv2

input_image_path = "./KneeXray/HH_1/4/"

img_list = os.listdir(input_image_path)
jpg_files = [file for file in img_list if file.endswith(".jpg")]

for img in jpg_files:
    image = cv2.imread(os.path.join(input_image_path, img))
    height, width = image.shape[:2]

    new_width = int(width * 0.85)
    new_height = int(height * 0.75)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    cropped_image = image[top:bottom, left:right]

    output_image_path = f"./KneeXray/HH_1_center_crop/4/{img}"
    cv2.imwrite(output_image_path, cropped_image)