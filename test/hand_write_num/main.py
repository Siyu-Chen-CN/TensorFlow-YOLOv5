# Classification
import os
import tensorflow as tf
import keras
from keras import datasets, optimizers, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#加载数据集
(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.batch(200)


# 准备结构
model = keras.Sequential()
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    with tf.GradientTape() as tape:
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, (-1, 28*28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()