import tensorflow as tf
import random
label = [0]*10 + [1]*10 + [2]*10
Features = [(random.uniform(0.7, 1), random.uniform(0, 0.2), random.uniform(0, 0.2)) for i in range(10)] + [(random.uniform(0.7, 1), random.uniform(0.7, 1), random.uniform(0, 0.2)) for i in range(10)] + [(random.uniform(0.7, 1), random.uniform(0.3, 0.6), random.uniform(0, 0.2)) for i in range(10)]
tenser1 = tf.convert_to_tensor(label)
tenser2 = tf.convert_to_tensor(Features)
print(tenser1.shape)
print(tenser2.shape)
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(3, input_shape=(3,))])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(tenser2, tenser1, epochs=200)
print(model.predict(tf.convert_to_tensor([(0.78, 0.1, 0.02), (0.91, 0.7, 0.02), (0.84, 0.3, 0.02)])))
