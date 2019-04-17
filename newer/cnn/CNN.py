from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds


tf.logging.set_verbosity(tf.logging.ERROR)

import math
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm

print(tf.__version__)

tf.enable_eager_execution()

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#separate the dataset

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

print("training {}".format(num_train_examples))
print("test:{}".format(test_dataset))

def normalize(images , lables):
    images = tf.cast(images,tf.float32)
    images /= 255
    return images, lables

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

