from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import pandas as pd


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images,test_images = train_images / 255.0, test_images / 255.0
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu,input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu))
model.summary()

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_images,test_labels)
print(test_acc)
