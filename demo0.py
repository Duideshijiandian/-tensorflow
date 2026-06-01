import tensorflow as tf
import random
label = [0]*10 + [1]*10 + [2]*10
Features = [(random.uniform(0.7, 1), random.uniform(0, 0.2), random.uniform(0, 0.2)) for i in range(10)] + [(random.uniform(0.7, 1), random.uniform(0.7, 1), random.uniform(0, 0.2)) for i in range(10)] + [(random.uniform(0.7, 1), random.uniform(0.3, 0.6), random.uniform(0, 0.2)) for i in range(10)]
tenser1 = tf.convert_to_tensor(label, dtype=tf.int64)
tenser2 = tf.convert_to_tensor(Features)
print(tenser1.shape)
print(tenser2.shape)
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(4, input_shape=(3,)),
     tf.keras.layers.ReLU(),
     tf.keras.layers.Dense(4),
     tf.keras.layers.ReLU(),
     tf.keras.layers.Dense(3)
     ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
a =model.fit(tenser2, tenser1, epochs=100)
lable1 = [0]*5 + [1]*5 + [2]*5
Features1 = [(random.uniform(0.7, 1), random.uniform(0, 0.2), random.uniform(0, 0.2)) for i in range(5)] + [(random.uniform(0.7, 1), random.uniform(0.7, 1), random.uniform(0, 0.2)) for i in range(5)] + [(random.uniform(0.7, 1), random.uniform(0.3, 0.6), random.uniform(0, 0.2)) for i in range(5)]
tenser3 = tf.convert_to_tensor(lable1, dtype=tf.int64)
tenser4 = tf.convert_to_tensor(Features1)
jieguo = tf.argmax(model.predict(tenser4), axis=1)
print(jieguo)
count = 0
for i in range(15):
    if jieguo[i] == tenser3[i]:
        count += 1
print("准确率为：", count/15)
print(a.history['loss'][-1])
