import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

  # ========== 模型 ==========
model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(256, activation='relu'),

      # BUG-1: 这一层激活函数换了，想想有没有问题？
      tf.keras.layers.Dense(128, activation='relu'),

      tf.keras.layers.Dense(10)
  ])

  # BUG-2: 学习率设得很大，想想会怎样？
model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  # 准备日志
import os
log_dir = os.path.join('logs', 'debug')
tensorboard = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      histogram_freq=1,
      update_freq='epoch'
  )

model.fit(
      train_images, train_labels,
      epochs=15,
      validation_data=(test_images, test_labels),
      callbacks=[tensorboard]
  )