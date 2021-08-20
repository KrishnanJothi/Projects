import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def PLACEHOLDER_FLOW(frames):
    return np.array(
        [[[x, y] for x in np.linspace(-1, 1, frames[0].shape[1])] for y in np.linspace(-1, 1, frames[0].shape[0])])


PLACEHOLDER_FLOW_VISUALIZATION = cv2.imread('resources/example_flow_visualization.png')


#
# Task 1
#
# Implement utility functions for flow visualization.


# TODO: Convert a flow map to a BGR image for visualisation. A flow map is a 2-channel 2D image with channel 1 and 2
#  depicting the portion flow in X and Y direction respectively.
def flowMapToBGR(flow_map):
    # Flow vector (X, Y) to angle
    magnitude, ang = cv2.cartToPolar(flow_map[:, :, 0], flow_map[:, :, 1], angleInDegrees=True)
    hsv = np.zeros(shape=(flow_map.shape[0], flow_map.shape[1], 3))
    # hsv[:, :, 0] = np.mod(ang / (2 * np.pi) + 1.0, 1.0)
    hsv[:, :, 0] = ((ang / 2) // 6) * 6
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    hsv[:, :, 1] = cv2.normalize(1 - hsv[:, :, 2], None, 0, 1, cv2.NORM_MINMAX)
    # Angle and vector size to HSV color
    bgr = cv2.cvtColor(hsv.astype('float32'), cv2.COLOR_HSV2RGB)
    return bgr


# TODO: Draw arrows depicting the provided `flow` on a 10x10 pixel grid.
#       You may use `cv2.arrowedLine(..)`.
def drawArrows(img, flow, arrow_color=(0, 255, 0)):
    out_img = img.copy()
    magnitude, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=False)
    # magnitude = cv2.normalize(magnitude, None, 0, 10, cv2.NORM_MINMAX)
    for i in range(0, flow.shape[0], 10):
        for j in range(0, flow.shape[1], 10):
            increment_x, increment_y = (10, 10)
            if i + 10 > flow.shape[0]:
                increment_y = flow.shape[0] - i
            if j + 10 > flow.shape[1]:
                increment_x = flow.shape[1] - j
            avg_magnitude = np.mean(magnitude[i: i + increment_y, j: j + increment_x])
            avg_angle = np.mean(ang[i: i + increment_y, j: j + increment_x])
            flow_start = (j, i)
            flow_end = (int(j + avg_magnitude * np.cos(avg_angle))
                        if int(j + avg_magnitude * np.cos(avg_angle)) > 0 else 0,
                        int(i + avg_magnitude * np.sin(avg_angle))
                        if int(i + avg_magnitude * np.sin(avg_angle)) > 0 else 0)
            out_img = cv2.arrowedLine(out_img, flow_start, flow_end, color=arrow_color, tipLength=0.2)
    return out_img


# Calculate the angular error of an estimated optical flow compared to ground truth
def calculateAngularError(estimated_flow, groundtruth_flow):
    nom = groundtruth_flow[:, :, 0] * estimated_flow[:, :, 0] + groundtruth_flow[:, :, 1] * estimated_flow[:, :, 1] + \
          1.0
    denom = np.sqrt((groundtruth_flow[:, :, 0] ** 2 + groundtruth_flow[:, :, 1] ** 2 + 1.0) * (
            estimated_flow[:, :, 0] ** 2 + estimated_flow[:, :, 1] ** 2 + 1.0))
    return (1.0 / (estimated_flow.shape[0] * estimated_flow.shape[1])) * np.sum(np.arccos(np.clip(nom / denom, 0, 1)))


# Load a flow map from a file
def load_FLO_file(filename):
    if os.path.isfile(filename) is False:
        print("file does not exist %r" % str(filename))
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    if magic != 202021.25:
        print('Magic number incorrect. .flo file is invalid')
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    # The float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1],
    # v[row0,col1], ..., In total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    # Some cleanup (remove cv-destroying large numbers)
    flow[np.sqrt(np.sum(flow ** 2, axis=2)) > 100] = 0
    return flow