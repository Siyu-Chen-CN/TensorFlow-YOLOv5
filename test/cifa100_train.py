import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.python.keras import layers, optimizers, Sequential
from tensorflow.python.tpu import datasets

tf.random.set_seed(1234)

def block(filterNum):
    return """
            layers.Conv2D({}, [3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D({}, [3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    """.format(filterNum)

# convLayers = []
# for i in channels:
#     convLayers.append(block(i))
# print(type(convLayers))
'''
conv_layers = [
    # unit 1
    layers.Conv2D(64, [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 2
    # layers.Conv2D(128, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.Conv2D(128, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 3
    # layers.Conv2D(256, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.Conv2D(256, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 4
    # layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # # unit 5
    # layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.Conv2D(512, [3, 3], padding="same", activation=tf.nn.relu),
    # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
]
'''

def parse_model(model_dict):
    for chin, chout, (m, args) in enumerate(model_dict['backbone']):  # from, number, module, args
        print(m, args)
        # m = eval(m) if isinstance(m, str) else m  # eval stringsa


def main():
    # channels = [64, 128, 256, 512, 512]
    # convNet = Sequential(eval(block(i)) for i in channels)
    # convNet.build(input_shape=[None, 32, 32, 3])
    # x = tf.random.normal([10, 32, 32, 3])
    # out = convNet(x)
    # print(out.shape)
    parse_model(model_dict)

if __name__ == '__main__':
    main()