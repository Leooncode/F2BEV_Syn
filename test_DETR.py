import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


bevfeature = "predictions/f2bev_conv_st_seg/bevfeatures/0.npy"
focal = "predictions/f2bev_conv_st_seg/predfull/focal/45.npy"
rgb_img = "F2BEV_code/data/SynWood/semantic_annotations/rgbLabels/00000_BEV.png"
bev = np.load(bevfeature)
f = np.load(focal)
rgb_img = cv2.imread(rgb_img)
print(rgb_img.shape)

# 获取每个像素的最大类别
seg_classes = np.argmax(f, axis=0)

# 定义颜色映射
colors = [
    [0, 0, 0],          # unlabeled
    [70, 70, 70],      # building
    [100, 40, 40],     # fence
    [55, 90, 80],      # other
    [220, 20, 60],     # pedestrian
    [153, 153, 153],   # pole
    [157, 234, 50],    # road line
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [107, 142, 35],    # vegetation
    [0, 0, 142],       # four-wheeler vehicle
    [102, 102, 156],   # wall
    [220, 220, 0],     # traffic sign
    [70, 130, 180],    # sky
    [81, 0, 81],       # ground
    [150, 100, 100],   # bridge
    [230, 150, 140],   # rail track
    [180, 165, 180],   # guard rail
    [250, 170, 30],    # traffic light
    [45, 60, 150],     # water
    [145, 170, 100],   # terrain
    [0, 0, 230],       # two-wheeler vehicle
    [110, 190, 160],   # static
    [170, 120, 50],    # dynamic
    [255, 255, 255],   # ego-vehicle
]

# 创建 RGB 图像
rgb_image = np.zeros((800, 800, 3), dtype=np.uint8)

# 填充颜色
for class_id in range(25):
    mask = seg_classes == class_id
    rgb_image[mask] = colors[class_id]

# 显示合并后的图像
plt.imshow(rgb_image)
plt.title('Combined Segmentation Result')
plt.axis('off')
plt.show()