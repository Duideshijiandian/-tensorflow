import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)
print(test_images.shape)
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(10)
     ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
a =model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
jieguo = tf.argmax(model.predict(test_images), axis=1)
count = 0
for i in range(10000):
    if jieguo[i] == test_labels[i]:
        count += 1
print("准确率为：", count/10000)
print(a.history['loss'][-1])

