import numpy as np
import cv2
import matplotlib.pyplot as plt

front_masek = './forward_looking_camera_model/masks/front.npy'
bev_mask = 'unity_data/bev_mask.npy'
t = np.load(front_masek)
b = np.load(bev_mask)
b = b.reshape(4, 1, 50, 50, 3)
b_plot = b[0, 0]
# print(b_plot.shape)
# plt.imshow(b_plot)
# plt.show()
