"""
MNIST with 2 digit numbers. 
"""

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets ('MNIST_data', one_hot = True)

import tensorflow as tf 
import numpy as np 
sess = tf.InteractiveSession () 

x = tf.placeholder (tf.float32, shape = [None, 1568])
y_ = tf.placeholder (tf.float32, shape = [None, 20])

W = tf.Variable (tf.zeros ([1568, 20]))
b = tf.Variable (tf.zeros ([20]))

sess.run (tf.global_variables_initializer())

y = tf.matmul (x, W) + b 

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
	)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #very slow 


for _ in range(2000):
  batch_raw = [mnist.train.next_batch(100), mnist.train.next_batch(100)]
  xm = np.concatenate((batch_raw[0][0], batch_raw[1][0]), axis = 1)
  ym = np.concatenate((batch_raw[0][1], batch_raw[1][1]), axis = 1)
  train_step.run(feed_dict={x: xm, y_: ym}) 

#note that we can replace any tensor with feed_dict, not only placeholders. 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


mti = mnist.test.images
mtl = mnist.test.labels
p = np.random.permutation(len(mti))
mtic = np.concatenate ((mti, mti[p]), axis = 1) 
mtlc = np.concatenate ((mtl, mtl[p]), axis = 1)

print(accuracy.eval(feed_dict={x: mtic, y_: mtlc}))
