"""
MNIST with 2 digit numbers. 
"""
import tensorflow as tf 
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets ('MNIST_data', one_hot = False)

sess = tf.InteractiveSession ()

x = tf.placeholder (tf.float32, shape = [None, 1568])
y_ = tf.placeholder (tf.int64)

W = tf.Variable (tf.zeros ([1568, 100]))
b = tf.Variable (tf.zeros ([100]))

sess.run (tf.global_variables_initializer())

y = tf.matmul (x, W) + b 


cross_entropy = tf.reduce_mean(
	tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = y)
	)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #very slow 

for _ in range(1000):
  batch_raw = [mnist.train.next_batch(100), mnist.train.next_batch(100)]
  xm = np.concatenate((batch_raw[0][0], batch_raw[1][0]), axis = 1)
  ym = batch_raw[0][1] * 10 + batch_raw[1][1]  

  train_step.run(feed_dict={x: xm, y_: ym}) 

correct_prediction = tf.equal(tf.argmax(y,1), y_)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mti = mnist.test.images
mtl = mnist.test.labels
p = np.random.permutation(len(mti))
mtic = np.concatenate ((mti, mti[p]), axis = 1) 
mtlc = mtl * 10 + mtl[p]

print(accuracy.eval(feed_dict={x: mtic, y_: mtlc}))