import tensorflow as tf
from keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60k, 28, 28]
# y: [60k]
(x, y) , _ = datasets.mnist.load_data()

# x: [0, 1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# print(tf.reduce_min(y), tf.reduce_max(y))

# 创建一个数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
# print('batch: ', sample[0].shape, sample[1].shape)

# 创建权值
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

for epoch in range(10): # iterate db for 10
    # h1 = x@w1 + b1
    for step, (x, y) in enumerate(train_db):  # for every batch
        # y: [128]
        x = tf.reshape(x, [-1, 28*28]) # x: [128, 28, 28] -> [128, 28*28]
        
        with tf.GradientTape() as tape:  # tf.variable
            h1 = x@w1 + b1 # [128, 786]@[786, 256] + [256]
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2 # [128, 256]@[256, 128] + [128]
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] -> [b, 10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            # [b ,10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)
        
        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 需要原地更新, 不可以使用 w1 = w1 - lr * grads[0]
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(step, 'loss:', float(loss))