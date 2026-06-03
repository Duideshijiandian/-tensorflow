import tensorflow as tf
base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base.trainable = False
model = tf.keras.Sequential([base,
                             tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(256, activation='relu'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = tf.image.resize(train_images, (96, 96))
test_images = tf.image.resize(test_images, (96, 96))
tf.keras.applications.mobilenet_v2.preprocess_input(train_images)
tf.keras.applications.mobilenet_v2.preprocess_input(test_images)
a = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
b = model.predict(test_images)
b = tf.argmax(b, axis=1)
count = 0
for i in range(10000):
    if b[i] == test_labels[i][0]:
        count += 1
print("准确率为：", count/10000)
