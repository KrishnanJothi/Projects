#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np

from flow_utils import *
from utils import *


#
# Task 3
#
# Load and use a pretrained model to estimate the optical flow of the same two frames as in Task 2.


# Load image frames
frames = [  cv2.imread("resources/frame1.png"),
            cv2.imread("resources/frame2.png")]

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("resources/groundTruthOF.flo")



# TODO: Load the model.
import sys

MODEL_PATH = "../PreTrainedModels/LiteFlowNet2-TF2"
sys.path.insert(0, MODEL_PATH)

import tensorflow.compat.v1 as tf
from model import LiteFlowNet2

tf.disable_eager_execution()
useSintelDatasetModel = True

# Create TF session
sess = tf.Session()
model = LiteFlowNet2(isSintel=useSintelDatasetModel)
tens1 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
tens2 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
out = model(tens1, tens2)

# Load model weights
saver = tf.train.Saver()
if useSintelDatasetModel:
    saver.restore(sess, MODEL_PATH + "/models/LiteFlowNet2_Sintel_model")
else:
    saver.restore(sess, MODEL_PATH + "/models/LiteFlowNet2_Kitti_model")



# TODO: Run model inference on the two frames.
import math

def pad_image(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1

    nh = int(math.ceil(h / 32.) * 32)
    nw = int(math.ceil(w / 32.) * 32)

    pad_image = np.zeros([nh, nw, c])
    pad_image[:h, :w] = image
    return pad_image

def prepareImage(image):
    return np.float32(np.expand_dims(pad_image(image[..., ::-1]), 0)) / 255.0

def computeFlow(frame1, frame2):
    h, w = frame1.shape[:2]
    inp1 = prepareImage(frame1)
    inp2 = prepareImage(frame2)

    # Apply model to provided frames
    flow = sess.run(out, feed_dict={tens1: inp1, tens2: inp2})[0, :h, :w, :]

    # Generate visualizations
    flow_color = flowMapToBGR(flow)
    flow_arrows = drawArrows(frame1, flow)
    return flow, flow_color, flow_arrows



# Infer the flow for the two sample images
flow, flow_color, flow_arrows = computeFlow(frames[0], frames[1])



# Create and show visualizations for the computed flow
plt.figure(figsize=(5, 8))
error = calculateAngularError(flow, flow_gt)
showImages([("LiteFlowNet2 flow - angular error: %.3f" % error, flow_color),
            ("LiteFlowNet2 field", flow_arrows) ], 1)



import matplotlib.animation as animation

camera_idx = 0
cap = cv2.VideoCapture(camera_idx)
while not cap.isOpened():
    if camera_idx >= 10:
        break
    camera_idx += 1
    cap = cv2.VideoCapture(camera_idx)
if not cap.isOpened():
    print("Camera Error: Not available")
else:
    success, lastFrame = cap.read()
    if success:
        fig = plt.figure(figsize=(9, 6))
        imgs = showImages([lastFrame] * 3, None, show_window_now=False)
        def update_imgs(i):
            global cap
            global imgs
            global lastFrame
            success, frame = cap.read()
            if not success:
                print("Camera Error: No frame available")
            _, flow_color, flow_arrows = computeFlow(lastFrame, frame)
            imgs[0].set_data(convertColorImagesBGR2RGB(lastFrame)[0])
            imgs[1].set_data(convertColorImagesBGR2RGB(flow_color)[0])
            imgs[1].set_data(convertColorImagesBGR2RGB(flow_arrows)[0])
            lastFrame = frame
            
        ani = animation.FuncAnimation(fig, update_imgs, frames=250, interval=40, repeat=False)
        plt.show()

    cap.release()
#==
