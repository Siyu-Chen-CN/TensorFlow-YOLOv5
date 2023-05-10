import tensorflow as tf
from tensorflow.python.keras import layers, Sequential

def cifa100():
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
    convNet = Sequential(conv_layers)

    fcNet = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),     # 全连接层
    ])

    convNet.build(input_shape=[None, 32, 32, 3])
    fcNet.build(input_shape=[None, 512])

    variables = convNet.trainable_variables + fcNet.trainable_variables

    return convNet, fcNet, variables
    

