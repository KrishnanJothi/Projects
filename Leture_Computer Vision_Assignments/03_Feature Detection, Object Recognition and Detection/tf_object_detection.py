#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#
# Task 4
#
# Load and use a pretrained `Object Detection` model from the `TensorFlow Model Garden`

# Download/cache required model and test data
import keras

from tf_utils import *

print('Download model...')
modelPath = download_model('20200713', 'centernet_hg104_1024x1024_coco17_tpu-32')
print(modelPath)

print('Download labels...')
labelsPath = download_labels('mscoco_label_map.pbtxt')
print(labelsPath)

print('Download test images...')
imagePaths = download_test_images(['image1.jpg', 'image2.jpg'])
print('\n'.join(imagePaths))



# Load the model

import tensorflow as tf
import os
from utils import *

print('Load model')

# TODO: Load the downloaded saved tensorflow model using `load_model(..)` from the keras module
savedModelPath = os.path.join(modelPath, "saved_model")
model = keras.models.load_model(savedModelPath)



# Load label map data (for plotting)

from object_detection.utils import label_map_util

print('Load labels')

category_index = label_map_util.create_category_index_from_labelmap(labelsPath, use_display_name=True)



# Run inference

import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg') # Reactivate GUI backend (deactivated by `import viz_utils`)

imgs = []
for image_path in imagePaths:
    print('Running inference for {}... '.format(image_path))
    img = cv2.imread(image_path)

    # TODO: Detect the objects in `img` using the loaded model
    # input needs to be a tensor
    input_tensor = tf.convert_to_tensor(img)

    # model expects a batch of images
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # TODO: Add bounding boxes and labels of the detected objects to `img` using `viz_utils.visualize_boxes_and_labels_on_image_array(..)`.
    #  Tipps: All required data is either already available or contained in the dict-like structure returned when applying the model to `img`.
    #         TensorFlow models return tensors with an additional batch dimension that is not required for visualitation, i.e. take only index[0] and convert it to a numpy array.
    #         TensorFlow returns detected classes as floats, but the visualization requires ints.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, : num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        img,
        detections['detection_boxes'], detections['detection_classes'], detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.3)

    imgs.append(img)
plt.figure(figsize=(13, 5))
showImages(imgs)
