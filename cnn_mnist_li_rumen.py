from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets ('MNIST_data', one_hot = False)

import tensorflow as tf 
import numpy as np 
sess = tf.InteractiveSession () 

x = tf.placeholder (tf.float32, shape = [2, None, 784])
y_ = tf.placeholder (tf.int64)

def weight_variable(shape): 
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.concat([tf.reshape(x[0],[-1,28,28,1]),tf.reshape(x[1],[-1,28,28,1])],2)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 14 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*14*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 100])
b_fc2 = bias_variable([100])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

keep_prob = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(2000):
  batch_raw = [mnist.train.next_batch(100), mnist.train.next_batch(100)]
  ym = batch_raw[0][1]*10+batch_raw[1][1]

  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:[batch_raw[0][0],batch_raw[1][0]], y_:ym, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

  train_step.run(feed_dict={x:[batch_raw[0][0],batch_raw[1][0]], y_:ym, keep_prob: 0.5})

mti = mnist.test.images
mtl = mnist.test.labels
p = np.random.permutation(len(mti))
mtic = [mti, mti[p]] 
mtlc = mtl * 10 + mtl[p]

print("test accuracy %g"%accuracy.eval(feed_dict={
    x:mtic, y_: mtlc, keep_prob: 1.0}))
