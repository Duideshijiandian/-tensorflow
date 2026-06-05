import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
a = model.fit(train_images, train_labels, 
              epochs=20, 
              validation_data=(test_images, test_labels), 
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, min_lr=0.00001)])
b = model.predict(test_images)
jieguo = tf.argmax(b, axis=1)
count = 0
for i in range(10000):
    if jieguo[i] == test_labels[i][0]:
        count += 1
print("准确率为：", count/10000)