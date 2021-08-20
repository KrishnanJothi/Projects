#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import cv2
from scipy.ndimage.filters import convolve as filter2
from matplotlib.pyplot import figure, draw, pause, gca

from flow_utils import *
from utils import *



#
# Task 2
#
# Implement Lucas-Kanade or Horn-Schunck Optical Flow.



# TODO: Implement Lucas-Kanade Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# kernel_size: kernel size
# eigen_threshold: threshold for determining if the optical flow is valid when performing Lucas-Kanade
# returns the Optical flow based on the Lucas-Kanade algorithm

def LucasKanadeFlow(frames, Ix, Iy, It, kernel_size, eigen_threshold = 0.01):

    '''
    prev = frames[0]
    next = frames[1]


    I1 = np.array(prev)
    I2 = np.array(next)
    S = np.shape(I1)

    # applying Gaussian filter to eliminate any noise
    I1_smooth = cv2.GaussianBlur(I1, (3,3), 0)
    I2_smooth = cv2.GaussianBlur(I2, (3,3), 0)

    # finding the good features
    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    features = cv2.goodFeaturesToTrack(I1, mask=None, **feature_params)


    feature = np.int0(features)

    u = v = np.nan * np.ones(S)

    for l in feature:
        j, i = l.ravel()
        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.

        IX = ([Ix[i - 1, j - 1], Ix[i, j - 1], Ix[i - 1, j - 1], Ix[i - 1, j], Ix[i, j], Ix[i + 1, j], Ix[i - 1, j + 1],
               Ix[i, j + 1], Ix[i + 1, j - 1]])  # The x-component of the gradient vector
        IY = ([Iy[i - 1, j - 1], Iy[i, j - 1], Iy[i - 1, j - 1], Iy[i - 1, j], Iy[i, j], Iy[i + 1, j], Iy[i - 1, j + 1],
               Iy[i, j + 1], Iy[i + 1, j - 1]])  # The Y-component of the gradient vector
        IT = ([It[i - 1, j - 1], It[i, j - 1], It[i - 1, j - 1], It[i - 1, j], It[i, j], It[i + 1, j], It[i - 1, j + 1],
               It[i, j + 1], It[i + 1, j - 1]])  # The XY-component of the gradient vector

        # Using the minimum least squares solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK))  # transpose of A
        LK = np.array(np.matrix.transpose(LK))

        A1 = np.dot(LK_T, LK)  # Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2, LK_T)

        (u[i, j], v[i, j]) = np.dot(A3, IT)  # we have the vectors with minimized square error



    return np.dstack((u,v))


# TODO: Implement Horn-Schunck Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# max_iterations: maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
# epsilon: the stopping criterion for the difference when performing the Horn-Schuck algorithm
# returns the Optical flow based on the Horn-Schunck algorithm
def HornSchunckFlow(frames, Ix, Iy, It, max_iterations = 1000, epsilon = 0.002):

    im1 = frames[0].astype(np.float32)
    im2 = frames[1].astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Derivatives
    [fx, fy, ft] = [Ix, Iy, It]

    HSKERN = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]], float)

    # Iteration to reduce error
    for _ in range(max_iterations):
        # %% Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
        # %% common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (epsilon ** 2 + fx ** 2 + fy ** 2)
        # %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    # compareGraphs(U, V, im2)

    return np.dstack((U,V))

# Load image frames
frames = [  cv2.imread("resources/frame1.png"),
            cv2.imread("resources/frame2.png")]

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("resources/groundTruthOF.flo")

# Grayscales
gray = [(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float64) for frame in frames]

# Get derivatives in X and Y
xdk = np.array([[-1.0, 1.0],[-1.0, 1.0]])
ydk = xdk.T
fx =  cv2.filter2D(gray[0], cv2.CV_64F, xdk) + cv2.filter2D(gray[1], cv2.CV_64F, xdk)
fy = cv2.filter2D(gray[0], cv2.CV_64F, ydk) + cv2.filter2D(gray[1], cv2.CV_64F, ydk)

# Get time derivative in time (frame1 -> frame2)
tdk1 =  np.ones((2,2))
tdk2 = tdk1 * -1
ft = cv2.filter2D(gray[0], cv2.CV_64F, tdk2) + cv2.filter2D(gray[1], cv2.CV_64F, tdk1)

# Ground truth flow
plt.figure(figsize=(5, 8))
showImages([("Groundtruth flow", flowMapToBGR(flow_gt)),
            ("Groundtruth field", drawArrows(frames[0], flow_gt)) ], 1, False)

# Lucas-Kanade flow
flow_lk = LucasKanadeFlow(gray, fx, fy, ft, [15, 15])
error_lk = calculateAngularError(flow_lk, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("LK flow - angular error: %.3f" % error_lk, flowMapToBGR(flow_lk)),
            ("LK field", drawArrows(frames[0], flow_lk)) ], 1, False)

# Horn-Schunk flow
flow_hs = HornSchunckFlow(gray, fx, fy, ft)
error_hs = calculateAngularError(flow_hs, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("HS flow - angular error %.3f" % error_hs, flowMapToBGR(flow_hs)),
            ("HS field", drawArrows(frames[0], flow_hs)) ], 1, False)

plt.show()
