from __future__ import print_function, division, absolute_import
import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_lables) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_lables = test_lables[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = create_model()
model.fit(train_images,train_labels,epochs=10,
          validation_data=(test_images,test_lables),
          callbacks=[cp_callback])

model = create_model()
loss, acc = model.evaluate(test_images,test_lables)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images,test_lables)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,verbose=1,save_weights_only=True,
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images,train_labels,
          epochs=50,callbacks=[cp_callback],
          validation_data=(test_images,test_lables),
          verbose=0)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_lables)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

#save entire
model = create_model()
model.fit(train_images, train_labels, epochs=5)

model.save('my_model.h5')

new_model = keras.models.load_model('my_model.h5')
new_model.summary()