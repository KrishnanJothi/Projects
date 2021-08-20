
import os
import pathlib
import tensorflow as tf

DOWNLOAD_CACHE_DIR = os.path.join('..', 'download_cache')


def download_model(model_date, model_name):
    global DOWNLOAD_CACHE_DIR
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True,
                                        cache_subdir=os.path.realpath(os.path.join(DOWNLOAD_CACHE_DIR, 'models')))
    return str(model_dir)


def download_labels(filename):
    global DOWNLOAD_CACHE_DIR
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False,
                                        cache_subdir=os.path.realpath(os.path.join(DOWNLOAD_CACHE_DIR, 'labels')))
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)


def download_test_images(filenames):
    global DOWNLOAD_CACHE_DIR
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    image_paths = []
    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False,
                                            cache_subdir=os.path.realpath(os.path.join(DOWNLOAD_CACHE_DIR, 'test_images')))
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))
    return image_paths
