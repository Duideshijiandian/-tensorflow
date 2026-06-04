# Day 13: 迁移学习 — MobileNetV2 + CIFAR-10
import tensorflow as tf

# ========== 准备数据 ==========
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = tf.image.resize(train_images, (96, 96))
test_images = tf.image.resize(test_images, (96, 96))
train_images = tf.keras.applications.mobilenet_v2.preprocess_input(train_images)
test_images = tf.keras.applications.mobilenet_v2.preprocess_input(test_images)

# ========== 阶段1：特征提取（冻结基座，只训分类头） ==========
base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("=== 阶段1：特征提取（基座冻结） ===")
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# ========== 阶段2：微调（解冻基座最后14层，小学习率） ==========
base.trainable = False
for layer in base.layers[-14:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 学习率降10倍
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("=== 阶段2：微调（最后14层解冻） ===")
model.fit(train_images, train_labels, epochs=5,
          validation_data=(test_images, test_labels),
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# ========== 评估 ==========
b = model.predict(test_images)
b = tf.argmax(b, axis=1)
count = 0
for i in range(10000):
    if b[i] == test_labels[i][0]:
        count += 1
print("准确率为：", count / 10000)
