# Classification
import tensorflow as tf
from keras import datasets, optimizers
from tensorflow.python import layers

(xs, ys), _ = datasets.mnist.load_data()
print("datasets:", xs.shape, ys.shape)

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs, ys))

for step, (x,y) in enumerate(db):
    print(step, x.shape, y, y.shape)