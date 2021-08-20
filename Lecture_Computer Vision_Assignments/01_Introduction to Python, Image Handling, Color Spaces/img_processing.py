#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 5
#
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter
matplotlib.use('TkAgg')

def showImage(img, show_window_now = True):
    # TODO: Convert the channel order of an image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt_img = plt.imshow(img)
    if show_window_now:
        plt.show()
    return plt_img

# Create an empty image (remove this when image loading works)
# img = np.zeros((10,10,3), dtype=np.uint8)


# TODO: Load the image "img/hummingbird_from_pixabay.png" with OpenCV (`cv2`) to the variable `img` and show it with `showImage(img)`.
img = cv2.imread("./img/hummingbird_from_pixabay.png")
showImage(img)

def imageStats(img):
    print("Image stats:")
    width, height, numChannels = img.shape

    print(" - Width: " + str(width))
    print(" - Height: " + str(height))
    print(" - Number Channels: " + str(numChannels))

# TODO: Print image stats of the hummingbird image.
imageStats(img)
# TODO: Change the color of the hummingbird to blue by swapping red and blue image channels.
blueHummingbird = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
showImage(blueHummingbird)

# TODO: Store the modified image as "blue_hummingbird.png" to your hard drive.
cv2.imwrite("./img/blue_hummingbird.png", blueHummingbird)


#
# Task 6
#

from matplotlib.widgets import Slider

# Prepare to show the original image and keep a reference so that we can update the image plot later.
plt.figure(figsize=(4, 6))
plt_img = showImage(img, False)

# TODO: Convert the original image to HSV color space.
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def img_update(hue_offset, img = img):
    print("Set hue offset to " + str(hue_offset))
    # TODO: Change the hue channel of the HSV image by `hue_offset`.
    # Mind that hue values in OpenCV range from 0-179.
    print(img)
    h, s, v = cv2.split(img)
    hnew = (h + hue_offset) % 179
    # hueChannel = 50
    img = cv2.merge([hnew, s, v])


    # TODO: Convert the modified HSV image back to RGB
    # and update the image in the plot window using `plt_img.set_data(img_rgb)`.
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt_img.set_data(img)

# img_update(10)
# Create an interactive slider for the hue value offset.
ax_hue = plt.axes([0.1, 0.04, 0.8, 0.06]) # x, y, width, height
slider_hue = Slider(ax=ax_hue, label='Hue', valmin=0, valmax=180, valinit=0, valstep=1)
slider_hue.on_changed(img_update)

# Now actually show the plot window
plt.show()
