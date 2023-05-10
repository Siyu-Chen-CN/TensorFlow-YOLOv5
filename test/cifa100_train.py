import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.python.keras import layers, Sequential, losses, optimizers
# from tensorflow.python.tpu import datasets
from keras import datasets

tf.random.set_seed(1234)

conv_layers = [
    # unit 1
    layers.Conv2D(64, [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 2
    layers.Conv2D(128, [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 3
    layers.Conv2D(256, [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 4
    layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 5
    layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
]


def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()  # (50000, 32, 32, 3) (50000, 1)
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)


def main():
    convNet = Sequential(conv_layers)
    
    # x = tf.random.normal([10, 32, 32, 3])
    # out = convNet(x)
    # print(out.shape)
    fcNet = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),     # 全连接层
    ])

    convNet.build(input_shape=[None, 32, 32, 3])
    fcNet.build(input_shape=[None, 512])
    optimizer = optimizers.adam_v2.Adam(lr=1e-4)

    variables = convNet.trainable_variables + fcNet.trainable_variables

    for epoch in range(50):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = convNet(x)  # [b, 1, 1, 512]
                out = tf.reshape(out, [-1, 512])  # flatten -> [b, 512]
                logics = fcNet(out)  # [b, 100]

                # calculate loss
                y_onehot = tf.one_hot(y, depth=100)
                loss = losses.categorical_crossentropy(y_onehot, logics, from_logits=True)
                loss = tf.reduce_mean(loss)
            
            grads= tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))


if __name__ == '__main__':
    main()