'''
Reference: https://ai.plainenglish.io/vggnet-with-tensorflow-transfer-learning-with-vgg16-included-7e5f6fa9479a
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
import time


(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
'''
x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
x_train = tf.repeat(x_train, 3, axis=3)
x_test = tf.repeat(x_test, 3, axis=3)
'''

base_model = tf.keras.applications.VGG16(include_top = False, input_shape = (32,32,3))
for layer in base_model.layers:
  layer.trainable = False

base_model.summary()

x = layers.Flatten()(base_model.output)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(10, activation = 'softmax')(x)

head_model = Model(inputs = base_model.input, outputs = predictions)



head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
start = time.time()
#head_model.summary()
history = head_model.fit(x_train, y_train, batch_size=128, epochs=20)
end = time.time()
print("Result of Cifar10 VGG16 Tensorflow:",end - start)

fig, axs = plt.subplots(2, 1, figsize=(15,15))

# axs[0].plot(history.history['loss'])
# axs[0].plot(history.history['val_loss'])
# axs[0].title.set_text('Training Loss vs Validation Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].legend(['Train','Val'])
#
# axs[1].plot(history.history['accuracy'])
# axs[1].plot(history.history['val_accuracy'])
# axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('Accuracy')
# axs[1].legend(['Train', 'Val'])


head_model.evaluate(x_test, y_test)
