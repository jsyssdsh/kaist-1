# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt


tf.set_random_seed(777)

x_data = [1, 2, 3, 5]
y_data = [1, 2, 3, 5]

w = tf.placeholder(tf.float32)

hypothesis = w * x_data
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

w_vals = []
c_vals = []

for i in range(-30, 50):
    curr_w = i*0.1
    curr_c = sess.run(cost, {w: curr_w})
    w_vals.append(curr_w)
    c_vals.append(curr_c)

plt.plot(w_vals, c_vals)
plt.show()
