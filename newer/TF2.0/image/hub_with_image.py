from __future__ import division,print_function,absolute_import
import tensorflow as tf


import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
from tensorflow import keras
import PIL.Image as  Image

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
IMAGE_SHAPE = (224,224)

classifier = tf.keras.Sequential([
    hub.keras_layer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
print(grace_hopper)