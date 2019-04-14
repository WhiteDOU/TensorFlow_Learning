import tensorflow as tf
import numpy as np
import time
tf.enable_eager_execution()
print(tf.add(1,2))
print(tf.test.is_gpu_available())

x=tf.random_uniform([3,3])
print(x)

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x,x)
    result = time.time() - start
    print("10 loops: {:0.2f}ms".format(1000 * result))

print("On cpu")
with tf.device("CPI:0"):
    x = tf.random_uniform([1000,1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000,1000])
        assert x.device.endswith("GOU:0")
        time_matmul(x)


#create a dataset
ds_tensprs = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])

import tempfile
_, filename = tempfile.mkstemp()

with open(filename,'w') as f:
    f.write("""Line1
    Line 2
    Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)
for x in ds_tensprs:
    print(x)

for x in ds_file:
    print(x)
