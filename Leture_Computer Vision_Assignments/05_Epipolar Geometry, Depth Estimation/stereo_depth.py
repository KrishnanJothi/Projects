#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt

from depth_utils import *
from utils import *



#
# Task 2
#
# 3D reconstruction - via depth estimation - from stereo disparity.

img1 = np.float32(cv2.imread('img/aloe1.png')) / 255
img2 = np.float32(cv2.imread('img/aloe2.png')) / 255

# Define search range.
search_left = 80
search_right = 20

h, w, _ = img1.shape

disparity = np.zeros(img1.shape[:2], np.float32)

# TODO: Calculate the `disparity` (horizontal distance) for each pixel of `img1` to the corresponding (best matching) pixel in `img2`.
#  For a pixel (x, y), use plain pixel color as simple feature descriptor and search for the closest match in `img2` within the range [x - search_left, x + search_right].
#  Disparity should be relative to the image width (`w`) and can be positive or negative, depending on whether the shift is to the left or right.
for y in range(h):
    if y % 100 == 0:
        print("line %d of %d" % (y, h))
    for x in range(w):
        # Get pixel color as descriptor.
        color = img1[y, x]

        # Get pixel color difference in curent range.
        color_difference = np.sum(np.abs(img2[y,max(0, x - search_left): min(x + search_right, w)] - color), axis=1)

        # Get the shift (pixel position difference).
        dx = color_difference.argmin() - search_left - min(0, x - search_left)

        # Normalize.
        disparity[y, x] = dx / w

# TODO: Convert to relative depth values from the `disparity`.
#  Pixels "closer to the camera" should be darker, pixels farther away should be brighter.
depth = 1 - np.abs(disparity)


# Visualization
h_small = h // 4
w_small = w // 4
depth_small = cv2.medianBlur(depth, 5)
depth_small = cv2.resize(depth_small, dsize=(w_small, h_small), interpolation=cv2.INTER_AREA)

plt.figure(figsize=(14, 4))
showImages([("img1", img1), ("img2", img2), ("depth from disparity", depth_small)], 5, show_window_now=False, padding=[.01, .01, .01, .1])

plot3D(depth_small, (2, 5), (0, 3), (2, 2))
plt.show()
