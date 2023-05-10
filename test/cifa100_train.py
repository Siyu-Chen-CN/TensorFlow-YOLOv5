import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.python.keras import losses, optimizers
# from tensorflow.python.tpu import datasets
from keras import datasets

from module import cifa100

tf.random.set_seed(1234)


def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()  # (50000, 32, 32, 3) (50000, 1)
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)


def main():
    convNet, fcNet, variables = cifa100()
    optimizer = optimizers.adam_v2.Adam(learning_rate=1e-4)
    
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
        total_num = 0
        total_correct = 0
        for step, (x, y) in enumerate(test_db):
            with tf.GradientTape() as tape:
                out = convNet(x)
                out  = tf.reshape(out, [-1, 512])
                logics = fcNet(out)
                prob = tf.nn.softmax(logics, 1)
                pred = tf.argmax(prob, 1)
                pred = tf.cast(pred, tf.int32)

                correct = tf.cast(tf.equal(pred, y), tf.int32)
                correct = tf.reduce_sum(correct)

                total_num += x.shape[0]
                total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

if __name__ == '__main__':
    main()