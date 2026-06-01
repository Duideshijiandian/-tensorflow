import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
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
a =model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

x = model.inputs[0]
y = model.layers[0].output
model1 = Model(inputs=x, outputs=y)
img = test_images[0].reshape(1, 28, 28, 1)
feature_map = model1.predict(img)

fig, ax = plt.subplots(4, 8, figsize=(12, 6))
ax = ax.ravel()

for i in range(32):
    fmap = feature_map[0, :, :, i]
    ax[i].imshow(fmap, cmap='viridis')
    ax[i].axis('off')
    ax[i].set_title(f'Filter {i}', fontsize=8)
plt.tight_layout()
plt.show()