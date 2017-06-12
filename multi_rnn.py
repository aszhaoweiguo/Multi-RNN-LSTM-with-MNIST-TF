# -*- coding:utf-8 -*-

"""
    Multi layer RNN with LSTM unit

    @ Accuracy:
"""

import tensorflow as tf
import create_data
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

mnist = input_data.read_data_sets("../MNIST/", one_hot=True)
print mnist.train.images.shape

# hyper-parameters
learning_rate = 0.001
batch_size = tf.placeholder(tf.int32)
input_size = 28
timestep_size = 28
hidden_size = 256
layer_num = 2
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

# construct model
X = tf.reshape(_X, [-1, 28, 28])
def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(hidden_size,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        reuse=tf.get_variable_scope().reuse)

def attn_cell():
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(),
                                         output_keep_prob=keep_prob)

mlstm_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(layer_num)])

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs = list()
state = init_state
with tf.variable_scope("RNN"):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]


# loss & optimizer
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0],
                                                       y: batch[1],
                                                       keep_prob: 1.0,
                                                       batch_size: _batch_size})
        print("Iter%d, step%d, training accuracy%g" % (mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0],
                                  y: batch[1],
                                  keep_prob: 0.5,
                                  batch_size: _batch_size})

# apply test set
print("Test accuracy %g" % (sess.run(accuracy, feed_dict={_X: mnist.test.images,
                                                          y: mnist.test.labels,
                                                          keep_prob: 1.0,
                                                          batch_size: mnist.test.images.shape[0]})))




























