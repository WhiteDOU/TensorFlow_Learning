from __future__ import division,absolute_import,print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas
tf.logging.set_verbosity(tf.logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))


l0 = tf.keras.layers.Dense(units=1 ,input_shape=[1])

model=tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finish Training")

plt.xlabel('epoch')
plt.ylabel('loss magnitude')
plt.plot(history.history['loss'])
plt.show()

print(model.predict([120]))
print(l0.get_weights())#g给出权重值

hidden=tf.keras.layers.Dense(units=2,input_shape=[3])
output=tf.keras.layers.Dense(units=1)
model=tf.keras.Sequential([hidden,output])



