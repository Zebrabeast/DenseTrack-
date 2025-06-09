import os
import numpy as np
import cv2

result_root = '/home/data/ly/PET/outputs/Drone/vis_dir_6/txt'
img_root = '/home/ly/yp/DroneCrowd/test_data/test_data/images'
save_root = '/home/ly/test_result'

# txt_lists = os.listdir(result_root)

txt_path = os.path.join(result_root, 'img017012_pred_positions.txt')
res_list = np.loadtxt(txt_path, dtype=int, delimiter=',')
# print(res)

image_path = os.path.join(img_root, 'img017012.jpg')

# 读取图像
image = cv2.imread(image_path)

# 准备绘制
box_size = 20
for res in res_list:
    x_center = res[1]
    y_center = res[2]
    top_left = (int(x_center - box_size/2), int(y_center-box_size/2))
    bottom_right = (int(x_center+box_size/2), int(y_center+box_size/2))

    # 在图像上绘制矩形框
    color = (0, 255, 0) # 绿色框
    thickness = 3

    image = cv2.rectangle(image, top_left, bottom_right,color,thickness)

# save
save_path = os.path.join(save_root, 'img017012_6.jpg')

cv2.imwrite(save_path, image)





