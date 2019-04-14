import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import  os
import pandas as pd
import  sklearn
import sys
import tensorflow as tf
from tensorflow import keras
import time
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

print("python",sys.version)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__,module.__version__)

assert sys.version_info >=(3,5)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


#image classification

fashion_minst=keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,Y_test) = fashion_minst.load_data()

x_valid,x_train=X_train_full[:5000],X_train_full[5000:]
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]

print(x_train.shape)
print(1)
plt.imshow(x_train[0],cmap="binary")
plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]])
print(x_valid.shape)
print(X_test.shape)

n_rows = 5
n_cols = 10
plt.figure(figsize=(n_cols*1.4,n_rows*1.6))
for row in range(n_rows):
    for col in range(n_rows):
        index=n_cols*row+col
        plt.subplot(n_rows , n_cols,index + 1)
        plt.imshow(x_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]])
    plt.show()


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

#examine

model.summary()
keras.utils.plot_model(model,"my_mnist_model.png",show_shapes=True)

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

history = model.fit(x_train,y_train,epochs=10,validation_data=(x_valid,y_valid))
plot_learning_curves(history)

history = model.fit(x_train,y_train,epochs==10,validation_data=(x_valid,y_valid))

model.evaluate(X_test,Y_test)

n_new = 10
X_new = X_test[:n_new]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = y_proba.argmax(axis=1)
print(y_pred)

y_pred = model.predict_classes(X_new)
print(y_pred)

y_proba.max(axis=1).round(2)

k = 3
top_k = np.argmax(-y_proba,axis=1)[:,:k]
print(top_k)

row_indices = np.tile(np.arange(len(top_k)),[k,1]).T

y_proba=[row_indices,top_k].round(2)




