# -*- coding: utf-8 -*-

import tensorflow as tf

tf.set_random_seed(777)

x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]

w = tf.Variable(tf.random_uniform([1], -1000, 1000))
b = tf.Variable(tf.random_uniform([1], -1000, 1000))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = w * x + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

feed = {x: x_data, y: y_data}

for step in range(2001):
    sess.run(train, feed)
    if step % 20 == 0:
        print(step, sess.run(cost, feed), sess.run(w), sess.run(b))


print(sess.run(hypothesis, {x: 4}))
print(sess.run(hypothesis, {x: 52956956}))
