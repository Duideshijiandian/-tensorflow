import tensorflow as tf
import numpy as np

def preprocess(x, y):
    x = tf.image.resize(x, (96, 96))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x, y

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
Dataset1 = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
Dataset2 = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
Dataset1 = Dataset1.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=5000).batch(64).prefetch(tf.data.AUTOTUNE)
Dataset2 = Dataset2.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base.trainable = False
model = tf.keras.Sequential([
                            tf.keras.layers.RandomFlip('horizontal'),
                            tf.keras.layers.RandomRotation(0.08),
                            base,
                            tf.keras.layers.GlobalAveragePooling2D(),
                            tf.keras.layers.Dense(256, activation='relu'),
                            tf.keras.layers.Dropout(0.3),
                            tf.keras.layers.Dense(10, activation='softmax')
                            ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(Dataset1, epochs=5, validation_data=Dataset2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

base.trainable = False
for layer in base.layers[-14:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(Dataset1, epochs=5, validation_data=Dataset2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

b = model.predict(Dataset2)
b = tf.argmax(b, axis=1)
count = 0
for i in range(10000):
    if b[i] == test_labels[i][0]:
        count += 1
print("准确率为：", count / 10000)