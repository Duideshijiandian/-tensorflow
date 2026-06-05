import numpy as np
import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
model = tf.keras.Sequential([
     tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'),
     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
     tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(100),
     tf.keras.layers.ReLU(),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10)])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
a = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
b = model.predict(test_images)
jieguo = tf.argmax(b, axis=1)
count = 0
for i in range(10000):
    if jieguo[i] == test_labels[i]:
        count += 1
print("准确率为：", count/10000)


