# -*- coding: utf-8 -*-

import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1, 1, 1, 1, 1],
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]  # 발산되는경우 있음 *****
y_data = [1., 2., 3., 4., 5.]

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.matmul(W, x_data)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)  # 0.05로하면 해결됨 ***
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
